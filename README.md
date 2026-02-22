# -Python-GUI-for-DO-IDO-with-2D-3D-visualization-step-by-step-execution-and-trace-replay.

Use this short `README.md`:

```md
# do-ido-gui

Python desktop GUI for Dandelion Optimizer (DO) and Improved Dandelion Optimizer (IDO), with 2D/3D visualization, step-by-step execution, and trace replay.

## Features
- DO and IDO implementations
- 2D and 3D function visualization
- Start / Step / Stop controls
- Agent + best-solution visualization
- Save/Load trace and Forward/Backward replay
- Convergence + diversity timeline

## Requirements
- Python 3
- numpy
- matplotlib
- tkinter (or `tk` package in conda)

## Install
```bash
conda install -n base -c conda-forge numpy matplotlib tk -y
```
or
```bash
python -m pip install numpy matplotlib
```

## Run
```bash
python HW2/hw2_track2a_do_ido_gui.py
```

## Generate output files
```bash
python HW2/hw2_track2a_do_ido_gui.py --batch-report --outdir HW2
```

Outputs are saved in `HW2/ioData/`.

## References
- Zhao et al., 2022 (DO)
- Duan et al., 2024 (IDO)
```
