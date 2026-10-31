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

## Method and metric entry points

| Paper object | Code | Command/configuration |
|---|---|---|
| Validation recovery, directed usefulness and provenance novelty | `src/stair/cli.py`: `run`; `core.allocation` | `beta_r`, `beta_a`, `eta`, novelty/usefulness thresholds |
| Mutual positive bids, capacity-one maximal matching and eligible-service debt | `core.allocation`, `core.matching` | `beta_d`, `debt_step`, `debt_max`; isolated clients accrue no debt |
| Fixed symmetric mixing and local SGD | `cli.run`, `Network` | `mixing` in `(0,0.5]`, `local_epochs`, `lr`, `batch_size` |
| Accuracy, bottom-decile accuracy, P95 missed eligible opportunities, harmful exchanges and model bytes | `cli.run` | `stair train/compare --config configs/default.json --data data/clients --output runs` |
| Random/age/no-debt/no-novelty controls and visualization | `cli.run`, `common.plot` | `stair compare ...`; `stair plot --output runs` |

`--input` images are uint8 NHWC (or NHW grayscale); preparation reserves 10% of each client's training samples for private validation and keeps supplied test samples separate. Only private validation probes update recovery/usefulness; test labels are used after training for metrics. `Network` is a small GroupNorm CNN. A provenance vector tracks per-origin local-update timestamps, and novelty is the fraction of origins newer at the sender. Failed transfers consume attempted model bytes, reset allocation debt, and leave model parameters unchanged. Harmful-exchange rate counts directed successful received updates whose private probe loss rises beyond `loss_tolerance`; the denominator is two per successful pair. P95 runs count missed eligible opportunities since service, pausing when ineligible, and include unfinished runs. Bottom decile averages the worst `ceil(0.1 * clients)` client accuracies.

All methods use the same rounds, input partitions, contacts, local optimizer and mixing rule. `model_byte_budget` is a hard upper bound, not an automatic matched-byte scheduler; do not claim a matched-byte comparison when output byte totals differ. CIFAR download and grouped NPZ imports are supported. FEMNIST/LEAF and CRAWDAD preprocessing, ResNet-18 and published D-FedAvg/MATCHA/RelaySum optimizers are not bundled. Internal controls and generated smoke images do not reproduce those benchmarks or the paper's reported numbers.

## Functional validation

Run the repository-contained check from the repository root after installing dependencies:

```bash
python scripts/validate.py
```

The check uses a fixed seed (17), small generated inputs, the real data preparation and method handlers, hand-computable semantic/metric cases, output serialization and plotting. Generated functional-check artifacts stay in ignored `runs/validation/`; they are not reference experiment results. No full training or benchmark evaluation was performed.

The validation also checks finite cross-entropy, gradients and actual updates of every CNN parameter, two-round client training, and checkpoint reload equality for the same input in `eval()` mode.

Actual validation environment: Windows, Python 3.13.5; numpy 2.1.3, scipy 1.15.3, matplotlib 3.10.0, torch 2.7.1+cu118 (CPU checks), torchvision 0.22.1. On 2026-11-01, `python scripts/validate.py` passed; the distributed implementation and new functional checks do not establish the paper's reported performance.
