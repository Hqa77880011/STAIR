# STAIR

PyTorch implementation of eligible-service debt for decentralized peer interaction allocation.

## Installation


Use Python 3.11 or newer. Create and activate a virtual environment, then install:

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell, then install:

```bash
python -m pip install -e .
``` For GPU execution, install matching PyTorch and torchvision builds using the [PyTorch selector](https://pytorch.org/get-started/locally/).


## Prepare clients and contacts

```bash
stair prepare --data data/clients
stair train --data data/clients --method stair --output runs
stair compare --data data/clients --output runs
stair plot --output runs
```

Preparation downloads [CIFAR-10/100](https://www.cs.toronto.edu/~kriz/cifar.html), creates paired train/test Dirichlet client distributions and holds out a private local validation probe. For [LEAF FEMNIST](https://github.com/TalwalkarLab/leaf), export the user-grouped arrays to an NPZ containing uint8 `x,y,client,test_x,test_y,test_client` and pass `--input`. Images are NHWC or NHW and client IDs are 0 through N-1. Every client needs train, validation and test data. Preserve original user splits in the exported arrays.

`--contacts contacts.json` imports a list indexed by communication round; each element lists undirected `[i,j]` edges. Discretize a [CRAWDAD Haggle contact trace](https://crawdad.org/) into this format before preparation. Without an imported trace, preparation creates community contacts with a toggling bridge and records them for all methods. Set `bridge_competitors` and `bridge_failure` for scarcity/partition controls.

## Allocation and learning

Recovery deficit uses private validation loss versus a 20-observation median and standard deviation. A version-vector provenance sketch estimates directed novelty as the fraction of source entries newer than the receiver's. Directed usefulness is an EMA of positive normalized immediate validation improvement after exchange. Positive bilateral bids, novelty/usefulness minima and contact availability define eligibility. Mutual-local-maximum matching uses common lexicographic edge priorities and repeats until maximal.

Debt rises only on eligible, unmatched rounds, stays unchanged without eligibility and resets on allocation even when transmission fails. External-information age resets only on successful nonredundant incorporation. A successful pair applies the fixed symmetric mixture **simultaneously** before local SGD. Freshness never changes the mixing coefficient. Logs distinguish allocation, transmission and incorporation. Local optimization is identical across methods.

`random` and `age` are contact-only controls; `no_debt` and `no_novelty` remove the respective mechanism. A compact two-convolution GroupNorm classifier supplies a directly runnable vision workload instead of the paper's ResNet-18. Thus learning accuracy reflects this backbone, while service/debt/mixing behavior follows the stated equations. External D-PSGD/MATCHA/RelaySum/DFedSAM algorithms are not aliased to these controls.

## Configuration and results

`configs/default.json` includes dataset, client/round counts, local SGD, Dirichlet alpha, mixing, bid coefficients, debt bounds, age normalization, usefulness EMA, novelty threshold, link success probability and byte ceiling. One attempted bilateral exchange costs two FP32 model vectors, including failed attempts; metadata bytes are excluded. A byte ceiling that would cut an allocated round raises an error. For matched-byte comparisons, choose run horizons from the saved cumulative-byte curves rather than asserting that identical rounds imply identical communication.

Each method saves `history.csv`, `trace.json`, final `models.pt`, `client_accuracy.json`, `config.json` and `metrics.json`. Metrics include mean/bottom-decile client accuracy, P95 missed eligible-opportunity run, harmful direction fraction among successful receiver updates and cumulative model bytes. Ineligible rounds pause an existing missed-opportunity run; allocated rounds end it. Harmful means immediate private validation loss exceeds its pre-mix value by `loss_tolerance`. Comparison CSV/JSON and PNG/SVG plots contain generated measurements.

## Multiple seeds

Run matched configurations with different seeds in separate directories, then aggregate the generated metric files:

```bash
stair-report --inputs runs/seed0/metrics.json runs/seed1/metrics.json --output outputs/seeds
```

Use actual metric paths from your runs, including method subdirectories where applicable. Keep dataset, split and experimental condition fixed within a report. Summary CSV/JSON gives per-method arithmetic means and sample standard deviations; a single run has no standard-deviation estimate.
