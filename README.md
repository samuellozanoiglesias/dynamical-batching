# Dynamical Batching vs. Uniform Batching — spiral reproduction

This repository contains a single self-contained notebook, `dynamical_batching.ipynb`, that reproduces the core
spiral experiment for one seed:

1. generates the spiral dataset (train and test sets);
2. builds an MLP in JAX;
3. trains it twice from the same initialisation and seed:
   **Uniform Batching (UB, A=1)** and **Dynamical Batching (DB, A=20, T=320)**;
4. computes, once per cycle, the three weight-level switching measures: switching-set size $|\mathcal S|/n$,
   switching-set persistence $J(\mathcal S_c,\mathcal S_{c-1})$ and the pair-averaged switching distance d;
5. saves one CSV per run and draws all figures.

## How to run

**Google Colab (recommended).** Download `dynamical_batching.ipynb`, open https://colab.research.google.com,
choose *File → Upload notebook*, and run all cells (*Runtime → Run all*). A CPU runtime is enough.

**Locally.** Python ≥ 3.10 with:

```bash
pip install jax optax numpy pandas matplotlib jupyter
jupyter notebook dynamical_batching.ipynb
```

A full run (both protocols, 125k steps each) takes well under a minute on a CPU. The whole training run is compiled
into a single XLA program (`jax.lax.scan` over steps and evaluation blocks, on-device batch sampling), and both
protocols share one compilation.

## Protocol

Classes take turns being the *focus class*. Each **oscillation** lasts $T$ steps and belongs to one class; during it the
focus-class weight ramps linearly $1 \to A \to 1$ (peak at T/2) while the other classes keep weight 1. Batch
proportions are $p_c \propto w_c$ and per-class counts are $\lfloor p_c B \rfloor$. A **cycle** is C consecutive
oscillations, one per class (C, T steps). Oscillations run for the whole training; A=1 gives uniform batching.

Class states $\theta_i(c)$ are the mean parameters over the central window [T/4, 3T/4) of class i's oscillation in
cycle c. The switching set is the smallest set of weights carrying 80% of the across-class variance of the
drift-corrected class states (drift taken from the previous cycle mean).

## Configuration

Everything is set in the `CONFIG` cell (Section 1). Main parameters:

| Key | Default | Meaning |
|---|---|---|
| `seed` | `0` | model initialisation and batch sampling (shared by UB and DB) |
| `data.points_per_class` | `100` | spiral points per class |
| `data.noise_std` | `0.2` | angular noise (rad) |
| `data.random_seed` | `10` | train-set seed (test set uses seed + 1) |
| `model.nn_width` / `model.num_hidden_layers` | `50` / `1` | MLP width and depth (ReLU) |
| `training.training_steps` | `125000` | optimiser steps |
| `training.batch_size` | `50` | batch size $B$ |
| `training.save_metrics_every_n_steps` | `40` | evaluation interval |
| `optimizer` | Adam, lr `2e-3` | optimiser settings |
| `oscillations.period_length` | `320` | $T$, steps per oscillation |
| `runs` | `{"UB": 1, "DB": 20}` | amplitude $A$ of each run |

The large-network setting uses `nn_width = 500` and `runs = {"UB": 1, "DB": 10}` with `period_length = 150`.
Results over several seeds are obtained by re-running with different values of `seed`.

## Outputs

Written to `results/` (and offered as `results.zip` on Colab):

| File | Content |
|---|---|
| `UB_A1_T320_seed0.csv`, `DB_A20_T320_seed0.csv` | one row per evaluation step (columns below) |
| `batch_composition.png` | class proportions in the batch vs. training step (first cycles) |
| `accuracy.png`, `loss.png` | train/test accuracy and cross-entropy loss |
| `decision_boundaries.png` | final decision boundaries with train (dots) and test (crosses) samples |
| `switching_abc.png`, `switching_abc.pdf` | switching-set size, persistence and switching distance vs. cycle |

CSV columns: `step`, `train_accuracy`, `test_accuracy`, `train_loss`, `test_loss`, `focus_class`, `focus_weight`,
`osc_phase`, `cycle`, `switch_frac`, `switch_persist`, `switching_distance`. The last four are filled only on the
row where the corresponding cycle ends.
