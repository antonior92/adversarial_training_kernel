# Adversarial Training Kernel

Research code for the paper:

> **Kernel Learning with Adversarial Features: Numerical Efficiency and Adaptive Regularization**

This repository implements adversarial training techniques for kernel-based regression models. The core idea is to learn robustness at the *kernel level* through an iterative reweighting scheme, rather than relying on input-level perturbations. This yields numerically efficient adversarial training with adaptive regularization.

## Overview

The `advkern` package provides:

- **Kernel Adversarial Training** -- An iterative algorithm that solves adversarially robust kernel ridge regression via sample reweighting (the "eta trick") with adaptive regularization.
- **Multiple Kernel Learning** -- Extends adversarial training to learn combinations of multiple kernels with adaptive weights.
- **Input-level Adversarial Training** -- PGD-based adversarial training that fine-tunes PyTorch models against Lp-norm bounded perturbations.
- **PGD Attack** -- Projected Gradient Descent for generating adversarial examples under L1, L2, and L-infinity constraints.
- **Kernel Functions** -- RBF, Linear, and Matern (1/2, 3/2, 5/2) kernels with both NumPy and PyTorch backends.

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/antonior92/adversarial_training_kernel.git
cd adversarial_training_kernel
pip install -r requirements.txt
```

### Dependencies

| Package       | Version |
|---------------|---------|
| PyTorch       | ~2.2.2  |
| NumPy         | ~1.26.4 |
| scikit-learn  | ~1.6.1  |
| matplotlib    | ~3.10.0 |
| pandas        | ~2.2.3  |
| pytest        | ~8.3.4  |
| ucimlrepo     | ~0.0.7  |

## Quick Start

The main estimators follow the scikit-learn API:

```python
from advkern.kernel_advtrain import AdvKernelTrain

# Fit an adversarially robust kernel ridge regression model
model = AdvKernelTrain(kernel='rbf', adv_radius=0.05)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Multiple Kernel Learning

```python
from advkern.multiple_kernel_advtrain import AdvMultipleKernelTrain

model = AdvMultipleKernelTrain(
    kernel=['rbf', 'matern3-2', 'linear'],
    adv_radius=0.05
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### PGD Attack

```python
import torch
from advkern.pgd import PGD

pgd = PGD(model, loss_fn=torch.nn.MSELoss(), p=torch.inf,
          adv_radius=8/255, step_size=2/255, nsteps=10)
X_adv = pgd(X, y)
```

## Project Structure

```
adversarial_training_kernel/
├── advkern/                        # Core Python package
│   ├── kernel_advtrain.py          # Kernel-level adversarial training
│   ├── multiple_kernel_advtrain.py # Multiple kernel learning variant
│   ├── inp_advtrain.py             # Input-level adversarial training (PGD)
│   ├── pgd.py                      # PGD attack implementation
│   ├── kernels.py                  # Kernel functions (RBF, Linear, Matern)
│   ├── data.py                     # Synthetic data generation
│   └── np2torch.py                 # NumPy/scikit-learn to PyTorch bridge
├── scripts/                        # Experiment scripts
│   ├── generate_figures.sh         # Master script to reproduce all figures
│   ├── onedim_curve_fitting.py     # 1D curve fitting experiments
│   ├── error_vs_sample_size.py     # Generalization vs. sample size
│   ├── error_vs_snr.py             # Performance vs. signal-to-noise ratio
│   ├── error_vs_delta.py           # Performance vs. adversarial radius
│   └── get_performance.py          # Benchmark on UCI datasets
├── tests/                          # Test suite
│   ├── test_advtrain.py            # Unit tests for adversarial training
│   └── test_benchmark.py           # Benchmark and PGD tests
├── styles/                         # Matplotlib styles and table formatting
├── print_tables.py                 # Generate LaTeX result tables
└── requirements.txt
```

## Reproducing Paper Results

To reproduce all figures and tables from the paper, run the master script from the repository root:

```bash
cd scripts
bash generate_figures.sh
```

This will:

1. **Fig. 1 & Fig. S.1** -- Generate 1D curve fitting plots for RBF and Matern kernels.
2. **Fig. S.3** -- Produce error vs. signal-to-noise ratio plots (linear and nonlinear settings).
3. **Fig. 1 (right) & Fig. S.2** -- Plot error vs. sample size across kernel types.
4. **Fig. 2 & Table 3** -- Evaluate performance on UCI regression datasets (diabetes, wine, abalone, pollution, crime).

Output figures are saved to `img/` and numerical results to `out/`.

### Individual Experiments

Each script can also be run independently. For example:

```bash
# 1D curve fitting with RBF kernel
python scripts/onedim_curve_fitting.py --kernel rbf --curve 2

# Benchmark on real datasets
python scripts/get_performance.py --csv_file out/results.csv
```

## Running Tests

```bash
export PYTHONPATH=.:$PYTHONPATH
pytest tests/
```

The test suite validates the adversarial training algorithms against convex optimization solutions (via `cvxpy`) and tests PGD attack correctness.

## Supported Kernels

| Kernel     | Key          | Parameters |
|------------|--------------|------------|
| RBF        | `rbf`        | `gamma`    |
| Linear     | `linear`     | --         |
| Matern 1/2 | `matern1-2` | `gamma`    |
| Matern 3/2 | `matern3-2` | `gamma`    |
| Matern 5/2 | `matern5-2` | `gamma`    |

## Key Parameters

| Parameter    | Description                                          |
|--------------|------------------------------------------------------|
| `adv_radius` | Adversarial perturbation budget (controls robustness)|
| `kernel`     | Kernel type (see table above)                        |
| `max_iter`   | Maximum iterations for the reweighting algorithm     |
| `utol`       | Convergence tolerance on dual coefficient updates    |
| `verbose`    | Print iteration progress                             |
