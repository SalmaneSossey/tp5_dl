# TP5 Knowledge Distillation

This repository contains the ENSIAS Deep Learning TP on knowledge distillation, from classical response-based KD to advanced feature-based and relational methods.

The main deliverable is the notebook [notebooks/TP5_Knowledge_Distillation.ipynb](./notebooks/TP5_Knowledge_Distillation.ipynb). The repository also includes modular support files in `src/`, but the notebook itself is standalone so it can run directly in VS Code or Colab.

## Project Overview

- Part 1 studies response-based KD and Attention Transfer on MNIST restricted to digits `0`, `1`, and `8`.
- Part 2 studies classical KD, FitNets, and RKD on CIFAR-10 restricted to `cat`, `dog`, `deer`, and `horse`.
- Figures are saved to `figures/` and result tables to `results/`.

## Environment Setup

Create and activate the local virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If pretrained torchvision weights cannot be downloaded, the notebook falls back to randomly initialized teacher backbones and clearly reports that fallback in the output.

## Run the Notebook

```powershell
jupyter notebook notebooks/TP5_Knowledge_Distillation.ipynb
```

The notebook is also structured to run in VS Code or Colab with relative paths only.

## Expected Structure

```text
.
├── notebooks/
│   └── TP5_Knowledge_Distillation.ipynb
├── src/
│   ├── __init__.py
│   ├── utils.py
│   ├── models.py
│   ├── kd_losses.py
│   ├── train_utils.py
│   ├── visualization.py
│   └── metrics.py
├── figures/
├── results/
├── reports/
│   └── REPORT.md
├── requirements.txt
└── README.md
```
