# STAIR

Implementation of eligible-service debt for decentralized peer interaction allocation.

## Installation

Use Python `>=3.11`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data preparation

Preparation downloads [CIFAR-10/100](https://www.cs.toronto.edu/~kriz/cifar.html) through torchvision, fixes client partitions and reserves local validation probes. Contact schedules are generated community contacts unless supplied with `--contacts contacts.json` (one undirected edge list per round). `--input clients.npz` accepts user-grouped `x,y,client,test_x,test_y,test_client` arrays.

```bash
stair prepare --config configs/default.json --data data/clients
```

## Training and execution

`configs/default.json` fixes seed `0`. Keep this config for every step; change `seed` there and use separate data/output directories for independent runs.

```bash
stair train --config configs/default.json --data data/clients --output runs
```

## Evaluation and comparisons

`compare` executes the available methods, including component controls, on the same prepared inputs.

```bash
stair compare --config configs/default.json --data data/clients --output runs
stair plot --output runs
```

## Outputs

`runs/<method>/` contains `history.csv`, `trace.json`, `models.pt`, `client_accuracy.json`, `config.json` and `metrics.json`. Comparisons write `runs/comparison.csv` and `runs/comparison.json`; plots go to `runs/figures/`.

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, torch 2.7.1+cu118, torchvision 0.22.1, scipy 1.15.3, matplotlib 3.10.0. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training, full evaluation and CI were not run for this review.
