#!/usr/bin/env python3

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import matplotlib

if "--batch" in sys.argv or "--batch-report" in sys.argv:
    matplotlib.use("Agg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import tkinter as tk
from tkinter import filedialog, messagebox, ttk



def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def clip_bounds(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lb), ub)


def to_1d_bounds(lb: float | np.ndarray, ub: float | np.ndarray, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    if np.isscalar(lb):
        lbv = np.full(dim, float(lb))
    else:
        lbv = np.asarray(lb, dtype=float).reshape(dim)
    if np.isscalar(ub):
        ubv = np.full(dim, float(ub))
    else:
        ubv = np.asarray(ub, dtype=float).reshape(dim)
    return lbv, ubv


def safe_float(v: str, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def safe_int(v: str, default: int) -> int:
    try:
        return int(float(v))
    except Exception:
        return default



def sphere(x: np.ndarray) -> float:
    return float(np.sum(x * x))


def rastrigin(x: np.ndarray) -> float:
    a = 10.0
    n = x.size
    return float(a * n + np.sum(x * x - a * np.cos(2.0 * math.pi * x)))


def ackley(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    n = x.size
    s1 = np.sum(x * x) / n
    s2 = np.sum(np.cos(2.0 * math.pi * x)) / n
    return float(-20.0 * np.exp(-0.2 * np.sqrt(s1)) - np.exp(s2) + 20.0 + math.e)


def rosenbrock(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return 0.0
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1.0) ** 2))


@dataclass
class ObjectiveSpec:
    name: str
    func: Callable[[np.ndarray], float]
    lb: float
    ub: float
    default_dim: int
    description: str


OBJECTIVES: Dict[str, ObjectiveSpec] = {
    "sphere": ObjectiveSpec(
        name="sphere",
        func=sphere,
        lb=-100.0,
        ub=100.0,
        default_dim=5,
        description="Sphere: f(x)=sum(x_i^2), global optimum at x=0",
    ),
    "rastrigin": ObjectiveSpec(
        name="rastrigin",
        func=rastrigin,
        lb=-5.12,
        ub=5.12,
        default_dim=5,
        description="Rastrigin: highly multimodal benchmark",
    ),
    "ackley": ObjectiveSpec(
        name="ackley",
        func=ackley,
        lb=-32.0,
        ub=32.0,
        default_dim=5,
        description="Ackley: multimodal benchmark",
    ),
    "rosenbrock": ObjectiveSpec(
        name="rosenbrock",
        func=rosenbrock,
        lb=-5.0,
        ub=10.0,
        default_dim=5,
        description="Rosenbrock valley benchmark",
    ),
}



def levy_flight(beta: float, dim: int, rng: np.random.Generator) -> np.ndarray:
    sigma_u = (
        math.gamma(1.0 + beta)
        * math.sin(math.pi * beta / 2.0)
        / (math.gamma((1.0 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0)))
    ) ** (1.0 / beta)
    u = rng.normal(0.0, sigma_u, size=dim)
    v = rng.normal(0.0, 1.0, size=dim)
    return u / (np.abs(v) ** (1.0 / beta) + 1e-12)



@dataclass
class DOConfig:
    pop_size: int = 30
    dim: int = 30
    lb: float = -100.0
    ub: float = 100.0
    iters: int = 300

    algo: str = "DO"
    levy_beta: float = 1.5
    cutoff: float = 1.5
    versoria_phi: float = 10.0

    seed: int = 0
    tol: float = 1e-12
    patience: int = 10**9


def population_diversity(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    center = np.mean(x, axis=0)
    d = np.linalg.norm(x - center, axis=1)
    return float(np.mean(d))


class DandelionStepper:

    def __init__(self, func: Callable[[np.ndarray], float], cfg: DOConfig):
        self.func = func
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.lb, self.ub = to_1d_bounds(cfg.lb, cfg.ub, cfg.dim)

        self.initialized = False
        self.stopped = False

        self.iteration = 0
        self.no_improve = 0

        self.x: np.ndarray = np.empty((0, 0), dtype=float)
        self.fit: np.ndarray = np.empty(0, dtype=float)

        self.best_x: np.ndarray = np.empty(0, dtype=float)
        self.best_f: float = float("inf")

        self.history_best: List[float] = []
        self.history_div: List[float] = []
        self.trace: List[Dict[str, object]] = []

        self._denom_q = max((cfg.iters * cfg.iters - 2 * cfg.iters + 1), 1)
        self._last_alpha = 0.0

    def _snapshot(self, prev_agents: Optional[np.ndarray]) -> Dict[str, object]:
        return {
            "iter": int(self.iteration),
            "agents": self.x.copy(),
            "prev_agents": None if prev_agents is None else prev_agents.copy(),
            "best_x": self.best_x.copy(),
            "best_f": float(self.best_f),
            "alpha": float(self._last_alpha),
            "diversity": float(self.history_div[-1]) if self.history_div else 0.0,
        }

    def initialize(self) -> Dict[str, object]:
        self.x = self.rng.uniform(self.lb, self.ub, size=(self.cfg.pop_size, self.cfg.dim))
        self.fit = np.array([self.func(row) for row in self.x], dtype=float)

        best_idx = int(np.argmin(self.fit))
        self.best_x = self.x[best_idx].copy()
        self.best_f = float(self.fit[best_idx])

        self.iteration = 0
        self.no_improve = 0
        self.stopped = False
        self.history_best = [self.best_f]
        self.history_div = [population_diversity(self.x)]

        self.trace = [self._snapshot(prev_agents=None)]
        self.initialized = True
        return self.trace[-1]

    def stop(self) -> None:
        self.stopped = True

    def is_finished(self) -> bool:
        if not self.initialized:
            return False
        if self.stopped:
            return True
        if self.iteration >= self.cfg.iters:
            return True
        if self.no_improve >= self.cfg.patience:
            return True
        return False

    def step_once(self) -> Optional[Dict[str, object]]:
        if not self.initialized:
            self.initialize()

        if self.is_finished():
            return None

        cfg = self.cfg
        t = self.iteration + 1
        T = cfg.iters

        prev = self.x.copy()

        alpha = float(self.rng.random() * ((t * t) / (T * T) - 2.0 * t / T + 1.0))
        self._last_alpha = alpha

        xs = self.rng.random(cfg.dim) * (self.ub - self.lb) + self.lb

        theta = float(self.rng.uniform(-math.pi, math.pi))
        r = 1.0 / math.exp(theta)
        vx = r * math.cos(theta)
        vy = r * math.sin(theta)

        ln_y = float(math.exp(self.rng.normal(0.0, 1.0)))
        q = (t * t) / self._denom_q - (2.0 * t) / self._denom_q + 1.0 + 1.0 / self._denom_q
        k = 1.0 - float(self.rng.random() * q)

        if float(self.rng.normal(0.0, 1.0)) < cfg.cutoff:
            x_rise = self.x + alpha * vx * vy * ln_y * (xs - self.x)
        else:
            x_rise = self.x * k

        x_rise = clip_bounds(x_rise, self.lb, self.ub)
        fit_rise = np.array([self.func(row) for row in x_rise], dtype=float)
        improved = fit_rise < self.fit
        self.x[improved] = x_rise[improved]
        self.fit[improved] = fit_rise[improved]

        x_mean = np.mean(self.x, axis=0)
        beta_t = self.rng.normal(0.0, 1.0, size=(cfg.pop_size, cfg.dim))
        x_desc = self.x - alpha * beta_t * (x_mean - alpha * beta_t * self.x)
        x_desc = clip_bounds(x_desc, self.lb, self.ub)
        fit_desc = np.array([self.func(row) for row in x_desc], dtype=float)
        improved = fit_desc < self.fit
        self.x[improved] = x_desc[improved]
        self.fit[improved] = fit_desc[improved]

        best_idx = int(np.argmin(self.fit))
        if float(self.fit[best_idx]) < self.best_f:
            self.best_f = float(self.fit[best_idx])
            self.best_x = self.x[best_idx].copy()

        delta = 2.0 * t / T

        if cfg.algo.strip().upper() == "IDO":
            fmin = float(np.min(self.fit))
            fave = float(np.mean(self.fit))
            denom = fave - fmin
            if abs(denom) < 1e-30:
                ai = np.zeros(cfg.pop_size, dtype=float)
            else:
                ai = (self.fit - fmin) / denom

            phi_m = np.where(
                ai <= 0.5,
                1.0 - 1.0 / (cfg.versoria_phi * (ai - 0.5) ** 2 + 2.0),
                1.0 / (cfg.versoria_phi * (ai - 0.5) ** 2 + 2.0),
            )
        else:
            phi_m = np.ones(cfg.pop_size, dtype=float)

        steps = np.vstack([levy_flight(cfg.levy_beta, cfg.dim, self.rng) for _ in range(cfg.pop_size)])
        x_land = phi_m[:, None] * self.best_x + steps * alpha * (self.best_x - self.x * delta)

        x_land = clip_bounds(x_land, self.lb, self.ub)
        fit_land = np.array([self.func(row) for row in x_land], dtype=float)

        improved = fit_land < self.fit
        self.x[improved] = x_land[improved]
        self.fit[improved] = fit_land[improved]

        best_idx = int(np.argmin(self.fit))
        if float(self.fit[best_idx]) + cfg.tol < self.best_f:
            self.best_f = float(self.fit[best_idx])
            self.best_x = self.x[best_idx].copy()
            self.no_improve = 0
        else:
            self.no_improve += 1

        self.iteration = t
        self.history_best.append(self.best_f)
        self.history_div.append(population_diversity(self.x))

        snap = self._snapshot(prev_agents=prev)
        self.trace.append(snap)
        return snap

    def run_all(self) -> Dict[str, object]:
        if not self.initialized:
            self.initialize()
        while not self.is_finished():
            nxt = self.step_once()
            if nxt is None:
                break
        return self.trace[-1]



def snapshot_to_jsonable(snapshot: Dict[str, object]) -> Dict[str, object]:
    payload = {
        "iter": int(snapshot["iter"]),
        "best_f": float(snapshot["best_f"]),
        "alpha": float(snapshot.get("alpha", 0.0)),
        "diversity": float(snapshot.get("diversity", 0.0)),
        "agents": np.asarray(snapshot["agents"], dtype=float).tolist(),
        "best_x": np.asarray(snapshot["best_x"], dtype=float).tolist(),
        "prev_agents": None,
    }
    if snapshot.get("prev_agents") is not None:
        payload["prev_agents"] = np.asarray(snapshot["prev_agents"], dtype=float).tolist()
    return payload


def snapshot_from_jsonable(snapshot: Dict[str, object]) -> Dict[str, object]:
    return {
        "iter": int(snapshot["iter"]),
        "best_f": float(snapshot["best_f"]),
        "alpha": float(snapshot.get("alpha", 0.0)),
        "diversity": float(snapshot.get("diversity", 0.0)),
        "agents": np.asarray(snapshot["agents"], dtype=float),
        "best_x": np.asarray(snapshot["best_x"], dtype=float),
        "prev_agents": None if snapshot.get("prev_agents") is None else np.asarray(snapshot["prev_agents"], dtype=float),
    }


def save_trace(
    filepath: str,
    trace: List[Dict[str, object]],
    meta: Dict[str, object],
) -> None:
    payload = {
        "format": "DO_IDO_TRACE_V1",
        "created_at": now_stamp(),
        "meta": meta,
        "trace": [snapshot_to_jsonable(s) for s in trace],
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_trace(filepath: str) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    with open(filepath, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("format") != "DO_IDO_TRACE_V1":
        raise ValueError("Unsupported trace format")
    trace = [snapshot_from_jsonable(s) for s in payload.get("trace", [])]
    meta = payload.get("meta", {})
    return trace, meta



class DOIDOApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        self.title("HW2 Track 2a: DO/IDO Visual Studio")
        self.geometry("1600x980")
        self.minsize(1300, 840)

        self.status_var = tk.StringVar(value="Ready")

        self.objective_var = tk.StringVar(value="sphere")
        self.algo_var = tk.StringVar(value="IDO")
        self.plot_mode_var = tk.StringVar(value="2D")

        self.dim_var = tk.StringVar(value="5")
        self.pop_var = tk.StringVar(value="30")
        self.iters_var = tk.StringVar(value="150")
        self.seed_var = tk.StringVar(value="0")

        self.cutoff_var = tk.StringVar(value="1.5")
        self.levy_beta_var = tk.StringVar(value="1.5")
        self.versoria_phi_var = tk.StringVar(value="10.0")
        self.patience_var = tk.StringVar(value="1000000")

        self.grid2d_var = tk.StringVar(value="70")
        self.grid3d_var = tk.StringVar(value="18")
        self.auto_delay_var = tk.StringVar(value="180")

        self.show_paths_var = tk.BooleanVar(value=True)
        self.show_labels_var = tk.BooleanVar(value=True)

        self.slider_rows: List[Dict[str, object]] = []
        self.control_guard = False
        self.fullscreen = False

        self.current_spec = OBJECTIVES[self.objective_var.get()]
        self.lb = self.current_spec.lb
        self.ub = self.current_spec.ub

        self.stepper: Optional[DandelionStepper] = None
        self.trace: List[Dict[str, object]] = []
        self.trace_meta: Dict[str, object] = {}
        self.trace_index = 0
        self.playback_mode = False
        self.auto_running = False

        self.landscape_cache: Dict[Tuple[object, ...], Dict[str, object]] = {}

        self.colorbar = None
        self.cursor_annotation = None

        self._build_menu()
        self._build_layout()
        self._build_toolbar()
        self._build_controls()
        self._build_plot()
        self._build_statusbar()

        self._sync_function_to_controls()
        self._rebuild_variable_controls(initial=True)
        self._draw_current_state()

        self.bind("<F11>", lambda _e: self.toggle_fullscreen())
        self.bind("<Escape>", lambda _e: self.set_fullscreen(False))

    def _build_menu(self) -> None:
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="New Run", command=self.on_start)
        file_menu.add_command(label="Save Trace", command=self.on_save_trace)
        file_menu.add_command(label="Load Trace", command=self.on_load_trace)
        file_menu.add_separator()
        file_menu.add_command(label="Export Plot PNG", command=self.on_export_png)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        run_menu = tk.Menu(menubar, tearoff=0)
        run_menu.add_command(label="Start", command=self.on_start)
        run_menu.add_command(label="Step", command=self.on_step)
        run_menu.add_command(label="Run Auto", command=self.on_run_auto)
        run_menu.add_command(label="Stop", command=self.on_stop)
        run_menu.add_separator()
        run_menu.add_command(label="Backward", command=self.on_backward)
        run_menu.add_command(label="Forward", command=self.on_forward)
        menubar.add_cascade(label="Run", menu=run_menu)

        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_radiobutton(label="2D View", variable=self.plot_mode_var, value="2D", command=self.on_mode_change)
        view_menu.add_radiobutton(label="3D View", variable=self.plot_mode_var, value="3D", command=self.on_mode_change)
        view_menu.add_separator()
        view_menu.add_command(label="Toggle Fullscreen", command=self.toggle_fullscreen)
        menubar.add_cascade(label="View", menu=view_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.on_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

    def _build_layout(self) -> None:
        self.toolbar_frame = ttk.Frame(self)
        self.toolbar_frame.pack(side=tk.TOP, fill=tk.X)

        self.main_pane = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        self.main_pane.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.plot_container = ttk.Frame(self.main_pane)
        self.controls_container = ttk.Frame(self.main_pane, width=420)

        self.main_pane.add(self.plot_container, weight=5)
        self.main_pane.add(self.controls_container, weight=2)

    def _build_toolbar(self) -> None:
        ttk.Button(self.toolbar_frame, text="Start", command=self.on_start).pack(side=tk.LEFT, padx=3, pady=4)
        ttk.Button(self.toolbar_frame, text="Step", command=self.on_step).pack(side=tk.LEFT, padx=3, pady=4)
        ttk.Button(self.toolbar_frame, text="Run Auto", command=self.on_run_auto).pack(side=tk.LEFT, padx=3, pady=4)
        ttk.Button(self.toolbar_frame, text="Stop", command=self.on_stop).pack(side=tk.LEFT, padx=3, pady=4)

        ttk.Separator(self.toolbar_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=3)

        ttk.Button(self.toolbar_frame, text="Backward", command=self.on_backward).pack(side=tk.LEFT, padx=3, pady=4)
        ttk.Button(self.toolbar_frame, text="Forward", command=self.on_forward).pack(side=tk.LEFT, padx=3, pady=4)

        ttk.Separator(self.toolbar_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=3)

        ttk.Button(self.toolbar_frame, text="Save Trace", command=self.on_save_trace).pack(side=tk.LEFT, padx=3, pady=4)
        ttk.Button(self.toolbar_frame, text="Load Trace", command=self.on_load_trace).pack(side=tk.LEFT, padx=3, pady=4)

        ttk.Separator(self.toolbar_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=3)

        ttk.Button(self.toolbar_frame, text="Minimize", command=self.iconify).pack(side=tk.LEFT, padx=3, pady=4)
        ttk.Button(self.toolbar_frame, text="Fullscreen", command=self.toggle_fullscreen).pack(side=tk.LEFT, padx=3, pady=4)
        ttk.Button(self.toolbar_frame, text="Close", command=self.on_close).pack(side=tk.LEFT, padx=3, pady=4)

    def _build_controls(self) -> None:
        top = ttk.Frame(self.controls_container)
        top.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        fn_frame = ttk.LabelFrame(top, text="Function & View")
        fn_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Label(fn_frame, text="Objective").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        obj_combo = ttk.Combobox(fn_frame, textvariable=self.objective_var, state="readonly", values=list(OBJECTIVES.keys()), width=16)
        obj_combo.grid(row=0, column=1, sticky="ew", padx=4, pady=3)
        obj_combo.bind("<<ComboboxSelected>>", lambda _e: self.on_objective_change())

        ttk.Label(fn_frame, text="View").grid(row=1, column=0, sticky="w", padx=4, pady=3)
        view_combo = ttk.Combobox(fn_frame, textvariable=self.plot_mode_var, state="readonly", values=["2D", "3D"], width=16)
        view_combo.grid(row=1, column=1, sticky="ew", padx=4, pady=3)
        view_combo.bind("<<ComboboxSelected>>", lambda _e: self.on_mode_change())

        ttk.Label(fn_frame, text="Dimension N").grid(row=2, column=0, sticky="w", padx=4, pady=3)
        dim_spin = ttk.Spinbox(fn_frame, from_=2, to=20, textvariable=self.dim_var, width=10, command=self.on_dim_change)
        dim_spin.grid(row=2, column=1, sticky="w", padx=4, pady=3)
        dim_spin.bind("<Return>", lambda _e: self.on_dim_change())
        dim_spin.bind("<FocusOut>", lambda _e: self.on_dim_change())

        fn_frame.columnconfigure(1, weight=1)

        param_frame = ttk.LabelFrame(top, text="Algorithm Parameters")
        param_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        row = 0
        row = self._add_labeled_entry(param_frame, row, "Algorithm", self.algo_var, is_combo=True, combo_values=["DO", "IDO"])
        row = self._add_labeled_entry(param_frame, row, "Population", self.pop_var)
        row = self._add_labeled_entry(param_frame, row, "Max Iterations", self.iters_var)
        row = self._add_labeled_entry(param_frame, row, "Seed", self.seed_var)
        row = self._add_labeled_entry(param_frame, row, "Cutoff", self.cutoff_var)
        row = self._add_labeled_entry(param_frame, row, "Levy beta", self.levy_beta_var)
        row = self._add_labeled_entry(param_frame, row, "Versoria phi", self.versoria_phi_var)
        row = self._add_labeled_entry(param_frame, row, "Patience", self.patience_var)

        vis_frame = ttk.LabelFrame(top, text="Visualization")
        vis_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Checkbutton(vis_frame, text="Show dashed movement paths", variable=self.show_paths_var, command=self._draw_current_state).pack(
            side=tk.TOP, anchor="w", padx=4, pady=2
        )
        ttk.Checkbutton(vis_frame, text="Show agent number labels", variable=self.show_labels_var, command=self._draw_current_state).pack(
            side=tk.TOP, anchor="w", padx=4, pady=2
        )

        vis_form = ttk.Frame(vis_frame)
        vis_form.pack(side=tk.TOP, fill=tk.X, padx=2, pady=2)
        self._add_labeled_entry(vis_form, 0, "2D grid size", self.grid2d_var)
        self._add_labeled_entry(vis_form, 1, "3D grid size", self.grid3d_var)
        self._add_labeled_entry(vis_form, 2, "Auto step delay (ms)", self.auto_delay_var)

        timeline_frame = ttk.LabelFrame(top, text="Trace Timeline (Extra)")
        timeline_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        self.timeline_scale = tk.Scale(
            timeline_frame,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            command=self.on_timeline_scrub,
            showvalue=True,
            length=340,
        )
        self.timeline_scale.pack(side=tk.TOP, fill=tk.X, padx=4, pady=4)

        slider_frame = ttk.LabelFrame(top, text="Variables: slider + textbox + axis checkbox")
        slider_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=6)

        self.slider_canvas = tk.Canvas(slider_frame, height=330)
        self.slider_scroll = ttk.Scrollbar(slider_frame, orient=tk.VERTICAL, command=self.slider_canvas.yview)
        self.slider_inner = ttk.Frame(self.slider_canvas)

        self.slider_inner.bind(
            "<Configure>",
            lambda _e: self.slider_canvas.configure(scrollregion=self.slider_canvas.bbox("all")),
        )

        self.slider_canvas.create_window((0, 0), window=self.slider_inner, anchor="nw")
        self.slider_canvas.configure(yscrollcommand=self.slider_scroll.set)

        self.slider_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.slider_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _build_plot(self) -> None:
        self.figure = Figure(figsize=(10.5, 8.2), dpi=100)
        self._create_plot_axes(need_3d=False)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_container)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.mpl_toolbar = NavigationToolbar2Tk(self.canvas, self.plot_container, pack_toolbar=False)
        self.mpl_toolbar.update()
        self.mpl_toolbar.pack(side=tk.TOP, fill=tk.X)

        self.canvas.mpl_connect("motion_notify_event", self.on_plot_mouse_move)

    def _create_plot_axes(self, need_3d: bool) -> None:
        self.figure.clf()
        gs = self.figure.add_gridspec(
            2,
            2,
            width_ratios=[28, 1],
            height_ratios=[1, 1],
            hspace=0.28,
            wspace=0.08,
        )
        if need_3d:
            self.ax_main = self.figure.add_subplot(gs[0, 0], projection="3d")
        else:
            self.ax_main = self.figure.add_subplot(gs[0, 0])
        self.ax_cbar = self.figure.add_subplot(gs[0, 1])
        self.ax_metrics = self.figure.add_subplot(gs[1, :])
        self.ax_metrics_right = None

        self.ax_cbar.clear()
        self.ax_cbar.set_axis_off()
        self.colorbar = None
        self.cursor_annotation = None

    def _build_statusbar(self) -> None:
        bar = tk.Frame(self, bd=1, relief=tk.SUNKEN)
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        left = tk.Label(bar, text="Status:", anchor="w", padx=8, pady=4)
        left.pack(side=tk.LEFT)
        self.status_label = tk.Label(bar, textvariable=self.status_var, anchor="w", padx=6, pady=4)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        try:
            bg = "#1f1f1f"
            fg = "#f5f5f5"
            bar.configure(bg=bg)
            left.configure(bg=bg, fg=fg)
            self.status_label.configure(bg=bg, fg=fg)
        except Exception:
            pass

    def _add_labeled_entry(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        var: tk.StringVar,
        is_combo: bool = False,
        combo_values: Optional[List[str]] = None,
    ) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=2)
        if is_combo:
            box = ttk.Combobox(parent, textvariable=var, state="readonly", values=combo_values or [])
            box.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
        else:
            ent = ttk.Entry(parent, textvariable=var)
            ent.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
        parent.columnconfigure(1, weight=1)
        return row + 1

    def _sync_function_to_controls(self) -> None:
        spec = OBJECTIVES[self.objective_var.get()]
        self.current_spec = spec
        self.lb, self.ub = spec.lb, spec.ub

        dim = safe_int(self.dim_var.get(), spec.default_dim)
        if dim < 2:
            dim = 2
        if dim > 20:
            dim = 20
        self.dim_var.set(str(dim))

        self.landscape_cache.clear()

    def _selected_required_axes(self) -> int:
        return 3 if self.plot_mode_var.get() == "3D" else 2

    def _selected_axes(self) -> List[int]:
        sel: List[int] = []
        for i, row in enumerate(self.slider_rows):
            if bool(row["axis_var"].get()):
                sel.append(i)
        return sel

    def _constant_vector_from_controls(self) -> np.ndarray:
        dim = len(self.slider_rows)
        x = np.zeros(dim, dtype=float)
        for i, row in enumerate(self.slider_rows):
            x[i] = float(row["slider"].get())
        return x

    def _rebuild_variable_controls(self, initial: bool = False) -> None:
        for w in self.slider_inner.winfo_children():
            w.destroy()
        self.slider_rows = []

        dim = safe_int(self.dim_var.get(), self.current_spec.default_dim)
        dim = max(2, min(20, dim))
        self.dim_var.set(str(dim))

        req = self._selected_required_axes()

        for i in range(dim):
            rowf = ttk.Frame(self.slider_inner)
            rowf.pack(side=tk.TOP, fill=tk.X, padx=4, pady=2)

            ttk.Label(rowf, text=f"x{i + 1}", width=4).pack(side=tk.LEFT)

            slider = tk.Scale(
                rowf,
                from_=self.lb,
                to=self.ub,
                orient=tk.HORIZONTAL,
                resolution=max((self.ub - self.lb) / 2000.0, 1e-4),
                length=160,
                command=lambda _v, idx=i: self.on_slider_change(idx),
            )
            slider.set(0.0)
            slider.pack(side=tk.LEFT, padx=4)

            val_var = tk.StringVar(value="0.0000")
            ent = ttk.Entry(rowf, textvariable=val_var, width=10)
            ent.pack(side=tk.LEFT, padx=4)
            ent.bind("<Return>", lambda _e, idx=i: self.on_entry_commit(idx))
            ent.bind("<FocusOut>", lambda _e, idx=i: self.on_entry_commit(idx))

            axis_var = tk.IntVar(value=1 if i < req else 0)
            chk = ttk.Checkbutton(rowf, text="Axis", variable=axis_var, command=lambda idx=i: self.on_axis_toggle(idx))
            chk.pack(side=tk.LEFT, padx=4)

            self.slider_rows.append(
                {
                    "frame": rowf,
                    "slider": slider,
                    "entry": ent,
                    "value_var": val_var,
                    "axis_var": axis_var,
                }
            )

        self._enforce_axis_selection(exact=False)
        self._update_slider_enabled_state()
        self._update_timeline_bounds()

        if not initial:
            self._draw_current_state()

    def _update_slider_enabled_state(self) -> None:
        for row in self.slider_rows:
            active = bool(row["axis_var"].get())
            row["slider"].configure(state=tk.DISABLED if active else tk.NORMAL)
            row["entry"].configure(state=("disabled" if active else "normal"))

    def _enforce_axis_selection(self, exact: bool = False) -> None:
        req = self._selected_required_axes()
        selected = self._selected_axes()

        if len(selected) > req:
            for idx in selected[req:]:
                self.slider_rows[idx]["axis_var"].set(0)
            selected = self._selected_axes()

        if exact and len(selected) < req:
            for i, row in enumerate(self.slider_rows):
                if i not in selected:
                    row["axis_var"].set(1)
                    selected.append(i)
                    if len(selected) == req:
                        break

        self._update_slider_enabled_state()

    def _validate_axis_selection_for_draw(self) -> bool:
        req = self._selected_required_axes()
        selected = self._selected_axes()
        if len(selected) != req:
            self.status_var.set(f"Select exactly {req} axis checkboxes for {self.plot_mode_var.get()} view.")
            return False
        return True

    def _build_config(self) -> DOConfig:
        dim = safe_int(self.dim_var.get(), self.current_spec.default_dim)
        dim = max(2, min(20, dim))

        cfg = DOConfig(
            pop_size=max(2, safe_int(self.pop_var.get(), 30)),
            dim=dim,
            lb=self.lb,
            ub=self.ub,
            iters=max(1, safe_int(self.iters_var.get(), 150)),
            algo=self.algo_var.get().strip().upper(),
            cutoff=safe_float(self.cutoff_var.get(), 1.5),
            levy_beta=safe_float(self.levy_beta_var.get(), 1.5),
            versoria_phi=safe_float(self.versoria_phi_var.get(), 10.0),
            seed=safe_int(self.seed_var.get(), 0),
            patience=max(1, safe_int(self.patience_var.get(), 10**9)),
        )
        return cfg

    def on_objective_change(self) -> None:
        self._sync_function_to_controls()
        self._rebuild_variable_controls(initial=False)
        self.status_var.set(f"Objective set to {self.objective_var.get()} with bounds [{self.lb}, {self.ub}]")

    def on_mode_change(self) -> None:
        self._enforce_axis_selection(exact=True)
        self.landscape_cache.clear()
        self._draw_current_state()

    def on_dim_change(self) -> None:
        self._sync_function_to_controls()
        self._rebuild_variable_controls(initial=False)
        self.status_var.set(f"Dimension changed to N={self.dim_var.get()}")

    def on_slider_change(self, idx: int) -> None:
        if self.control_guard:
            return
        row = self.slider_rows[idx]
        v = float(row["slider"].get())
        row["value_var"].set(f"{v:.4f}")
        self.landscape_cache.clear()
        self._draw_current_state(overlay_only=False)

    def on_entry_commit(self, idx: int) -> None:
        row = self.slider_rows[idx]
        v = safe_float(row["value_var"].get(), float(row["slider"].get()))
        v = max(self.lb, min(self.ub, v))

        self.control_guard = True
        row["slider"].set(v)
        row["value_var"].set(f"{v:.4f}")
        self.control_guard = False

        self.landscape_cache.clear()
        self._draw_current_state(overlay_only=False)

    def on_axis_toggle(self, idx: int) -> None:
        req = self._selected_required_axes()
        selected = self._selected_axes()

        if len(selected) > req:
            self.slider_rows[idx]["axis_var"].set(0)
            self.status_var.set(f"Only {req} axis checkboxes are allowed in {self.plot_mode_var.get()} mode.")

        self._update_slider_enabled_state()
        self._draw_current_state(overlay_only=False)

    def on_start(self) -> None:
        if not self._validate_axis_selection_for_draw():
            return

        self.auto_running = False
        cfg = self._build_config()
        spec = OBJECTIVES[self.objective_var.get()]

        self.stepper = DandelionStepper(spec.func, cfg)
        self.stepper.initialize()

        self.trace = self.stepper.trace
        self.trace_index = 0
        self.playback_mode = False
        self.trace_meta = {
            "objective": spec.name,
            "algo": cfg.algo,
            "config": dataclasses.asdict(cfg),
            "plot_mode": self.plot_mode_var.get(),
            "axes": self._selected_axes(),
            "notes": "Generated by GUI Start button",
        }

        self._update_timeline_bounds()
        self.timeline_scale.set(0)
        self._draw_current_state(overlay_only=False)

        self.status_var.set(
            f"Run initialized: {cfg.algo} on {spec.name} | iter=0/{cfg.iters} | best_f={self.trace[0]['best_f']:.6e}"
        )

    def on_step(self) -> None:
        if self.playback_mode:
            self.on_forward()
            return

        if self.stepper is None:
            self.on_start()
            if self.stepper is None:
                return

        snap = self.stepper.step_once()
        if snap is None:
            self.status_var.set("Run finished or stopped.")
            self._draw_current_state(overlay_only=True)
            return

        self.trace = self.stepper.trace
        self.trace_index = len(self.trace) - 1
        self._update_timeline_bounds()
        self.timeline_scale.set(self.trace_index)
        self._draw_current_state(overlay_only=True)

        self.status_var.set(
            f"Step iter={snap['iter']}/{self.stepper.cfg.iters} | best_f={snap['best_f']:.6e} | div={snap['diversity']:.4f}"
        )

    def on_run_auto(self) -> None:
        if self.playback_mode:
            self.status_var.set("Auto-run is disabled in playback mode.")
            return
        if self.stepper is None:
            self.on_start()
            if self.stepper is None:
                return

        self.auto_running = True
        self._auto_step_loop()

    def _auto_step_loop(self) -> None:
        if not self.auto_running:
            return

        if self.stepper is None:
            self.auto_running = False
            return

        snap = self.stepper.step_once()
        if snap is None:
            self.auto_running = False
            self.status_var.set("Auto-run finished.")
            self._draw_current_state(overlay_only=True)
            return

        self.trace = self.stepper.trace
        self.trace_index = len(self.trace) - 1
        self._update_timeline_bounds()
        self.timeline_scale.set(self.trace_index)
        self._draw_current_state(overlay_only=True)

        delay = max(20, safe_int(self.auto_delay_var.get(), 180))
        self.after(delay, self._auto_step_loop)

    def on_stop(self) -> None:
        self.auto_running = False
        if self.stepper is not None:
            self.stepper.stop()
        self.status_var.set("Execution stopped by user.")

    def on_backward(self) -> None:
        if not self.trace:
            self.status_var.set("No trace available.")
            return

        if self.trace_index > 0:
            self.trace_index -= 1
            self.timeline_scale.set(self.trace_index)
            self.playback_mode = True
            self._draw_current_state(overlay_only=True)
            snap = self.trace[self.trace_index]
            self.status_var.set(f"Playback iter={snap['iter']} | best_f={snap['best_f']:.6e}")

    def on_forward(self) -> None:
        if not self.trace:
            self.status_var.set("No trace available.")
            return

        if self.trace_index < len(self.trace) - 1:
            self.trace_index += 1
            self.timeline_scale.set(self.trace_index)
            self.playback_mode = True
            self._draw_current_state(overlay_only=True)
            snap = self.trace[self.trace_index]
            self.status_var.set(f"Playback iter={snap['iter']} | best_f={snap['best_f']:.6e}")
        else:
            self.status_var.set("Already at the latest snapshot.")

    def on_timeline_scrub(self, value: str) -> None:
        if not self.trace:
            return
        idx = safe_int(value, self.trace_index)
        idx = max(0, min(idx, len(self.trace) - 1))
        if idx == self.trace_index:
            return

        self.trace_index = idx
        self.playback_mode = True
        self._draw_current_state(overlay_only=True)

    def on_save_trace(self) -> None:
        if not self.trace:
            messagebox.showwarning("Save Trace", "No trace to save.")
            return

        default_dir = os.path.join(os.getcwd(), "ioData")
        ensure_dir(default_dir)
        default_name = f"trace_{self.objective_var.get()}_{self.algo_var.get()}_{now_stamp()}.json"
        path = filedialog.asksaveasfilename(
            title="Save trace JSON",
            initialdir=default_dir,
            initialfile=default_name,
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return

        meta = dict(self.trace_meta)
        meta["axes"] = self._selected_axes()
        meta["plot_mode"] = self.plot_mode_var.get()
        meta["saved_trace_index"] = self.trace_index

        save_trace(path, self.trace, meta)
        self.status_var.set(f"Trace saved: {path}")

    def on_load_trace(self) -> None:
        path = filedialog.askopenfilename(
            title="Load trace JSON",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return

        try:
            trace, meta = load_trace(path)
        except Exception as exc:
            messagebox.showerror("Load Trace", f"Failed to load trace:\n{exc}")
            return

        if not trace:
            messagebox.showwarning("Load Trace", "Trace file is empty.")
            return

        self.auto_running = False
        self.playback_mode = True
        self.stepper = None
        self.trace = trace
        self.trace_meta = meta
        self.trace_index = 0

        obj_name = str(meta.get("objective", self.objective_var.get()))
        if obj_name in OBJECTIVES:
            self.objective_var.set(obj_name)
            self._sync_function_to_controls()

        cfg_meta = meta.get("config", {})
        dim = safe_int(str(cfg_meta.get("dim", self.dim_var.get())), safe_int(self.dim_var.get(), 5))
        self.dim_var.set(str(max(2, min(dim, 20))))
        self._rebuild_variable_controls(initial=False)

        axes = meta.get("axes")
        req = self._selected_required_axes()
        if isinstance(axes, list) and len(axes) >= req:
            for i, row in enumerate(self.slider_rows):
                row["axis_var"].set(1 if i in axes[:req] else 0)
        self._enforce_axis_selection(exact=True)

        self._update_timeline_bounds()
        self.timeline_scale.set(0)
        self._draw_current_state(overlay_only=False)

        snap = self.trace[self.trace_index]
        self.status_var.set(f"Loaded trace ({len(self.trace)} steps): iter={snap['iter']} best_f={snap['best_f']:.6e}")

    def on_export_png(self) -> None:
        default_dir = os.path.join(os.getcwd(), "ioData")
        ensure_dir(default_dir)
        default_name = f"plot_{self.objective_var.get()}_{now_stamp()}.png"
        path = filedialog.asksaveasfilename(
            title="Export plot to PNG",
            initialdir=default_dir,
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
        )
        if not path:
            return

        self.figure.savefig(path, dpi=180)
        self.status_var.set(f"Plot exported: {path}")

    def on_about(self) -> None:
        msg = (
            "HW2 Track 2a GUI\n\n"
            "Implements Dandelion Optimizer (DO) and Improved Dandelion Optimizer (IDO)\n"
            "with step-by-step visualization, 2D/3D function view, and trace playback."
        )
        messagebox.showinfo("About", msg)

    def on_close(self) -> None:
        self.auto_running = False
        self.destroy()

    def set_fullscreen(self, enabled: bool) -> None:
        self.fullscreen = bool(enabled)
        self.attributes("-fullscreen", self.fullscreen)

    def toggle_fullscreen(self) -> None:
        self.set_fullscreen(not self.fullscreen)

    def _ensure_axes(self) -> None:
        mode = self.plot_mode_var.get()

        need_3d = mode == "3D"
        currently_3d = hasattr(self.ax_main, "zaxis")
        if need_3d == currently_3d:
            return

        self._create_plot_axes(need_3d=need_3d)

    def _landscape_cache_key(self) -> Optional[Tuple[object, ...]]:
        if not self._validate_axis_selection_for_draw():
            return None

        mode = self.plot_mode_var.get()
        axes = self._selected_axes()
        const = self._constant_vector_from_controls()

        req = self._selected_required_axes()
        keep = [const[i] for i in range(const.size) if i not in axes[:req]]

        if mode == "2D":
            res = max(20, safe_int(self.grid2d_var.get(), 70))
        else:
            res = max(6, safe_int(self.grid3d_var.get(), 18))

        key = (
            mode,
            self.objective_var.get(),
            safe_int(self.dim_var.get(), self.current_spec.default_dim),
            tuple(axes[:req]),
            tuple(round(v, 5) for v in keep),
            res,
        )
        return key

    def _compute_landscape(self) -> Optional[Dict[str, object]]:
        key = self._landscape_cache_key()
        if key is None:
            return None
        if key in self.landscape_cache:
            return self.landscape_cache[key]

        spec = OBJECTIVES[self.objective_var.get()]
        mode = self.plot_mode_var.get()
        axes = self._selected_axes()
        const = self._constant_vector_from_controls()

        if mode == "2D":
            i, j = axes[0], axes[1]
            res = max(20, safe_int(self.grid2d_var.get(), 70))
            xg = np.linspace(self.lb, self.ub, res)
            yg = np.linspace(self.lb, self.ub, res)
            xx, yy = np.meshgrid(xg, yg)
            zz = np.empty_like(xx)

            base = const.copy()
            for r in range(res):
                for c in range(res):
                    x = base.copy()
                    x[i] = xx[r, c]
                    x[j] = yy[r, c]
                    zz[r, c] = spec.func(x)

            data = {"mode": "2D", "axes": (i, j), "xx": xx, "yy": yy, "zz": zz}

        else:
            i, j, k = axes[0], axes[1], axes[2]
            res = max(6, safe_int(self.grid3d_var.get(), 18))
            grid = np.linspace(self.lb, self.ub, res)

            xs: List[float] = []
            ys: List[float] = []
            zs: List[float] = []
            fs: List[float] = []

            base = const.copy()
            for xv in grid:
                for yv in grid:
                    for zv in grid:
                        x = base.copy()
                        x[i] = xv
                        x[j] = yv
                        x[k] = zv
                        xs.append(xv)
                        ys.append(yv)
                        zs.append(zv)
                        fs.append(spec.func(x))

            data = {
                "mode": "3D",
                "axes": (i, j, k),
                "xs": np.asarray(xs, dtype=float),
                "ys": np.asarray(ys, dtype=float),
                "zs": np.asarray(zs, dtype=float),
                "fs": np.asarray(fs, dtype=float),
            }

        self.landscape_cache[key] = data
        return data

    def _draw_current_state(self, overlay_only: bool = False) -> None:
        self._ensure_axes()

        if not overlay_only:
            self._draw_landscape_background()

        self._draw_agents_overlay()
        self._draw_metrics_panel()
        self.canvas.draw_idle()

    def _draw_landscape_background(self) -> None:
        self.ax_main.clear()

        land = self._compute_landscape()
        if land is None:
            self.ax_cbar.clear()
            self.ax_cbar.set_axis_off()
            self.ax_main.set_title("Select required axis checkboxes")
            return

        mode = land["mode"]

        self.colorbar = None
        self.ax_cbar.clear()
        self.ax_cbar.set_axis_on()

        if mode == "2D":
            xx = land["xx"]
            yy = land["yy"]
            zz = land["zz"]
            im = self.ax_main.pcolormesh(xx, yy, zz, shading="auto", cmap="viridis")
            self.colorbar = self.figure.colorbar(im, cax=self.ax_cbar)
            self.colorbar.set_label("f(x)")

            i, j = land["axes"]
            self.ax_main.set_xlabel(f"x{i + 1}")
            self.ax_main.set_ylabel(f"x{j + 1}")
            self.ax_main.set_title(
                f"{self.objective_var.get()} landscape (2D slice) | fixed other vars by sliders"
            )

            if self.cursor_annotation is None:
                self.cursor_annotation = self.ax_main.annotate(
                    "",
                    xy=(0, 0),
                    xytext=(10, 10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75),
                )
                self.cursor_annotation.set_visible(False)

        else:
            xs = land["xs"]
            ys = land["ys"]
            zs = land["zs"]
            fs = land["fs"]

            sc = self.ax_main.scatter(xs, ys, zs, c=fs, cmap="viridis", s=8, alpha=0.32)
            self.colorbar = self.figure.colorbar(sc, cax=self.ax_cbar)
            self.colorbar.set_label("f(x)")

            i, j, k = land["axes"]
            self.ax_main.set_xlabel(f"x{i + 1}")
            self.ax_main.set_ylabel(f"x{j + 1}")
            self.ax_main.set_zlabel(f"x{k + 1}")
            self.ax_main.set_title("3D variable space (color = function value)")

    def _current_snapshot(self) -> Optional[Dict[str, object]]:
        if not self.trace:
            return None
        idx = max(0, min(self.trace_index, len(self.trace) - 1))
        return self.trace[idx]

    def _planned_total_iters(self) -> int:
        if self.stepper is not None:
            return max(1, int(self.stepper.cfg.iters))
        cfg_meta = self.trace_meta.get("config", {}) if isinstance(self.trace_meta, dict) else {}
        if isinstance(cfg_meta, dict) and "iters" in cfg_meta:
            return max(1, safe_int(str(cfg_meta.get("iters", 1)), 1))
        if self.trace:
            return max(1, int(self.trace[-1].get("iter", 1)))
        return 1

    def _draw_agents_overlay(self) -> None:
        snap = self._current_snapshot()
        if snap is None:
            return

        if not self._validate_axis_selection_for_draw():
            return

        mode = self.plot_mode_var.get()
        axes = self._selected_axes()

        agents = np.asarray(snap["agents"], dtype=float)
        best_x = np.asarray(snap["best_x"], dtype=float)

        prev_agents = snap.get("prev_agents")
        if prev_agents is None and self.trace_index > 0:
            prev_agents = self.trace[self.trace_index - 1]["agents"]

        if mode == "2D":
            i, j = axes[0], axes[1]

            xs = agents[:, i]
            ys = agents[:, j]
            self.ax_main.scatter(xs, ys, s=44, c="white", edgecolors="black", linewidths=0.7, zorder=4)

            if self.show_labels_var.get():
                for idx, (xv, yv) in enumerate(zip(xs, ys), start=1):
                    self.ax_main.text(xv, yv, str(idx), fontsize=7, color="black", zorder=5)

            self.ax_main.scatter(
                [best_x[i]],
                [best_x[j]],
                marker="*",
                s=230,
                c="red",
                edgecolors="black",
                linewidths=1.0,
                zorder=6,
                label="best",
            )

            if self.show_paths_var.get() and prev_agents is not None:
                prev_agents = np.asarray(prev_agents, dtype=float)
                for n in range(min(prev_agents.shape[0], agents.shape[0])):
                    self.ax_main.plot(
                        [prev_agents[n, i], agents[n, i]],
                        [prev_agents[n, j], agents[n, j]],
                        linestyle="--",
                        linewidth=0.8,
                        color="#ff5a36",
                        alpha=0.70,
                        zorder=3,
                    )
            elif self.show_paths_var.get():
                self.ax_main.text(
                    0.02,
                    0.98,
                    "Dashed paths appear after first Step",
                    transform=self.ax_main.transAxes,
                    fontsize=8,
                    va="top",
                    ha="left",
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.35),
                    zorder=7,
                )

        else:
            i, j, k = axes[0], axes[1], axes[2]

            xs = agents[:, i]
            ys = agents[:, j]
            zs = agents[:, k]

            self.ax_main.scatter(xs, ys, zs, s=32, c="white", edgecolors="black", linewidths=0.6, zorder=5)
            if self.show_labels_var.get():
                for idx, (xv, yv, zv) in enumerate(zip(xs, ys, zs), start=1):
                    self.ax_main.text(xv, yv, zv, str(idx), fontsize=7, zorder=6)

            self.ax_main.scatter(
                [best_x[i]],
                [best_x[j]],
                [best_x[k]],
                marker="*",
                s=250,
                c="red",
                edgecolors="black",
                linewidths=1.0,
                zorder=7,
            )

            if self.show_paths_var.get() and prev_agents is not None:
                prev_agents = np.asarray(prev_agents, dtype=float)
                for n in range(min(prev_agents.shape[0], agents.shape[0])):
                    self.ax_main.plot(
                        [prev_agents[n, i], agents[n, i]],
                        [prev_agents[n, j], agents[n, j]],
                        [prev_agents[n, k], agents[n, k]],
                        linestyle="--",
                        linewidth=0.75,
                        color="#ff5a36",
                        alpha=0.70,
                        zorder=4,
                    )
            elif self.show_paths_var.get():
                self.ax_main.text2D(
                    0.02,
                    0.98,
                    "Dashed paths appear after first Step",
                    transform=self.ax_main.transAxes,
                    fontsize=8,
                    va="top",
                    ha="left",
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.35),
                    zorder=7,
                )

    def _draw_metrics_panel(self) -> None:
        self.ax_metrics.clear()
        if self.ax_metrics_right is not None:
            try:
                self.ax_metrics_right.remove()
            except Exception:
                pass
            self.ax_metrics_right = None

        if not self.trace:
            self.ax_metrics.set_title("Convergence & Diversity")
            self.ax_metrics.set_xlabel("Iteration")
            self.ax_metrics.set_ylabel("Best fitness")
            return

        iters = [int(s["iter"]) for s in self.trace]
        best = [float(s["best_f"]) for s in self.trace]
        div = [float(s.get("diversity", 0.0)) for s in self.trace]

        l_best = self.ax_metrics.plot(iters, best, color="tab:blue", linewidth=1.8, label="best fitness")
        self.ax_metrics.scatter(iters, best, color="tab:blue", s=14, alpha=0.8)
        self.ax_metrics.set_ylabel("Best fitness", color="tab:blue")
        self.ax_metrics.tick_params(axis="y", labelcolor="tab:blue")

        self.ax_metrics_right = self.ax_metrics.twinx()
        l_div = self.ax_metrics_right.plot(iters, div, color="tab:orange", linewidth=1.5, label="population diversity")
        self.ax_metrics_right.scatter(iters, div, color="tab:orange", s=14, alpha=0.8)
        self.ax_metrics_right.set_ylabel("Population diversity", color="tab:orange")
        self.ax_metrics_right.tick_params(axis="y", labelcolor="tab:orange")

        current_iter = int(self.trace[self.trace_index]["iter"])
        l_step = self.ax_metrics.axvline(current_iter, color="tab:red", linestyle="--", linewidth=1.0, label="current step")

        x_right = max(self._planned_total_iters(), max(iters) if iters else 1)
        self.ax_metrics.set_xlim(0, x_right)
        self.ax_metrics_right.set_xlim(0, x_right)

        if best:
            y_min = min(best)
            y_max = max(best)
            if abs(y_max - y_min) < 1e-12:
                span = max(1.0, abs(y_max) * 0.1 + 1e-6)
                self.ax_metrics.set_ylim(y_min - span, y_max + span)
        if div:
            y_min_d = min(div)
            y_max_d = max(div)
            if abs(y_max_d - y_min_d) < 1e-12:
                span_d = max(1.0, abs(y_max_d) * 0.1 + 1e-6)
                self.ax_metrics_right.set_ylim(y_min_d - span_d, y_max_d + span_d)

        if len(iters) <= 1:
            self.ax_metrics.text(
                0.02,
                0.88,
                "Press Step or Run Auto to build curves",
                transform=self.ax_metrics.transAxes,
                fontsize=9,
                va="top",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.7),
            )

        self.ax_metrics.set_title("Convergence + diversity timeline")
        self.ax_metrics.set_xlabel("Iteration")
        self.ax_metrics.grid(True, alpha=0.25)
        legend_handles = l_best + l_div + [l_step]
        legend_labels = [h.get_label() for h in legend_handles]
        self.ax_metrics.legend(legend_handles, legend_labels, loc="best", fontsize=8)

    def _update_timeline_bounds(self) -> None:
        max_idx = max(0, len(self.trace) - 1)
        self.timeline_scale.configure(to=max_idx)

    def on_plot_mouse_move(self, event) -> None:
        if event.inaxes != self.ax_main:
            if self.cursor_annotation is not None:
                self.cursor_annotation.set_visible(False)
                self.canvas.draw_idle()
            return

        if self.plot_mode_var.get() != "2D":
            return
        if event.xdata is None or event.ydata is None:
            return

        if not self._validate_axis_selection_for_draw():
            return

        axes = self._selected_axes()
        i, j = axes[0], axes[1]

        x = self._constant_vector_from_controls()
        x[i] = float(event.xdata)
        x[j] = float(event.ydata)

        spec = OBJECTIVES[self.objective_var.get()]
        fv = spec.func(x)

        if self.cursor_annotation is not None:
            self.cursor_annotation.xy = (event.xdata, event.ydata)
            self.cursor_annotation.set_text(f"f={fv:.4e}")
            self.cursor_annotation.set_visible(True)

        snap = self._current_snapshot()
        if snap is not None:
            self.status_var.set(
                f"Cursor: x{i + 1}={event.xdata:.4f}, x{j + 1}={event.ydata:.4f}, f={fv:.4e} | "
                f"iter={snap['iter']} best_f={snap['best_f']:.4e}"
            )

        self.canvas.draw_idle()



def run_batch_report(base_outdir: str) -> Dict[str, object]:
    ensure_dir(base_outdir)
    io_dir = os.path.join(base_outdir, "ioData")
    ensure_dir(io_dir)

    experiments = [
        {"name": "sphere", "algo": "DO", "dim": 30, "pop": 30, "iters": 200, "seed": 0},
        {"name": "sphere", "algo": "IDO", "dim": 30, "pop": 30, "iters": 200, "seed": 0},
        {"name": "rastrigin", "algo": "DO", "dim": 30, "pop": 40, "iters": 250, "seed": 0},
        {"name": "rastrigin", "algo": "IDO", "dim": 30, "pop": 40, "iters": 250, "seed": 0},
        {"name": "ackley", "algo": "DO", "dim": 30, "pop": 35, "iters": 220, "seed": 2},
        {"name": "ackley", "algo": "IDO", "dim": 30, "pop": 35, "iters": 220, "seed": 0},
    ]

    rows: List[Dict[str, object]] = []

    for exp in experiments:
        spec = OBJECTIVES[exp["name"]]
        cfg = DOConfig(
            pop_size=exp["pop"],
            dim=exp["dim"],
            lb=spec.lb,
            ub=spec.ub,
            iters=exp["iters"],
            algo=exp["algo"],
            seed=exp["seed"],
            cutoff=1.5,
            levy_beta=1.5,
            versoria_phi=10.0,
        )
        stepper = DandelionStepper(spec.func, cfg)
        stepper.initialize()
        stepper.run_all()
        final = stepper.trace[-1]

        row = {
            "example": exp["name"],
            "algo": exp["algo"],
            "dim": exp["dim"],
            "pop": exp["pop"],
            "iters": exp["iters"],
            "seed": exp["seed"],
            "best_f": float(final["best_f"]),
            "iters_ran": int(final["iter"]),
        }
        rows.append(row)

    summary = {
        "created_at": now_stamp(),
        "description": "HW2 Track 2a batch output summary",
        "results": rows,
    }

    summary_path = os.path.join(io_dir, "hw2_track2a_batch_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    sample_spec = OBJECTIVES["sphere"]
    sample_cfg = DOConfig(pop_size=16, dim=5, lb=sample_spec.lb, ub=sample_spec.ub, iters=40, algo="IDO", seed=7)
    sample_stepper = DandelionStepper(sample_spec.func, sample_cfg)
    sample_stepper.initialize()
    sample_stepper.run_all()

    sample_meta = {
        "objective": "sphere",
        "algo": "IDO",
        "config": dataclasses.asdict(sample_cfg),
        "plot_mode": "2D",
        "axes": [0, 1],
        "notes": "Sample trace generated by --batch-report",
    }
    sample_trace_path = os.path.join(io_dir, "hw2_track2a_sample_trace.json")
    save_trace(sample_trace_path, sample_stepper.trace, sample_meta)

    fig = Figure(figsize=(8, 6), dpi=140)
    ax = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    res = 80
    xg = np.linspace(sample_spec.lb, sample_spec.ub, res)
    yg = np.linspace(sample_spec.lb, sample_spec.ub, res)
    xx, yy = np.meshgrid(xg, yg)
    zz = np.empty_like(xx)
    for r in range(res):
        for c in range(res):
            v = np.zeros(sample_cfg.dim, dtype=float)
            v[0] = xx[r, c]
            v[1] = yy[r, c]
            zz[r, c] = sample_spec.func(v)

    pc = ax.pcolormesh(xx, yy, zz, shading="auto", cmap="viridis")
    fig.colorbar(pc, ax=ax, pad=0.01, label="f(x)")

    last = sample_stepper.trace[-1]
    agents = np.asarray(last["agents"], dtype=float)
    best = np.asarray(last["best_x"], dtype=float)
    ax.scatter(agents[:, 0], agents[:, 1], c="white", edgecolors="black", s=28)
    ax.scatter([best[0]], [best[1]], marker="*", s=180, c="red", edgecolors="black")
    ax.set_title("Sample final state on Sphere (x1-x2 slice)")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    its = [int(s["iter"]) for s in sample_stepper.trace]
    best_hist = [float(s["best_f"]) for s in sample_stepper.trace]
    div_hist = [float(s["diversity"]) for s in sample_stepper.trace]
    ax2.plot(its, best_hist, label="best fitness", color="tab:blue")
    ax2.plot(its, div_hist, label="diversity", color="tab:orange")
    ax2.set_title("Convergence and diversity")
    ax2.set_xlabel("iteration")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best", fontsize=8)

    png_path = os.path.join(io_dir, "hw2_track2a_sample_plot.png")
    fig.savefig(png_path, dpi=140)

    return {
        "summary_path": summary_path,
        "sample_trace_path": sample_trace_path,
        "sample_plot_path": png_path,
        "summary": summary,
    }



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HW2 Track 2a DO/IDO GUI")
    p.add_argument("--batch", action="store_true", help="Alias to run headless report batch output")
    p.add_argument("--batch-report", action="store_true", help="Run headless benchmark and trace generation")
    p.add_argument("--outdir", default=".", help="Base output directory for batch mode")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.batch or args.batch_report:
        out = run_batch_report(args.outdir)
        print("Batch report generated:")
        print(f"- Summary: {out['summary_path']}")
        print(f"- Sample trace: {out['sample_trace_path']}")
        print(f"- Sample plot: {out['sample_plot_path']}")
        print("\nBenchmark summary:")
        for row in out["summary"]["results"]:
            print(
                f"{row['example']:10s} {row['algo']:4s} dim={row['dim']:2d} pop={row['pop']:2d} "
                f"iters={row['iters']:3d} seed={row['seed']:2d} best_f={row['best_f']:.6e}"
            )
        return

    app = DOIDOApp()
    app.mainloop()


if __name__ == "__main__":
    main()
