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

Preparation downloads [CIFAR-10/100](https://cave.cs.toronto.edu/kriz/cifar.html) through torchvision, fixes client partitions and reserves local validation probes. Contact schedules are generated community contacts unless supplied with `--contacts contacts.json` (one undirected edge list per round). `--input clients.npz` accepts user-grouped `x,y,client,test_x,test_y,test_client` arrays.

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

## Components and metrics

| Component / metric | Code | Command/configuration |
|---|---|---|
| Validation recovery, directed usefulness and provenance novelty | `src/stair/cli.py`: `run`; `core.allocation` | `beta_r`, `beta_a`, `eta`, novelty/usefulness thresholds |
| Mutual positive bids, capacity-one maximal matching and eligible-service debt | `core.allocation`, `core.matching` | `beta_d`, `debt_step`, `debt_max`; isolated clients accrue no debt |
| Fixed symmetric mixing and local SGD | `cli.run`, `Network` | `mixing` in `(0,0.5]`, `local_epochs`, `lr`, `batch_size` |
| Accuracy, bottom-decile accuracy, P95 missed eligible opportunities, harmful exchanges and model bytes | `cli.run` | `stair train/compare --config configs/default.json --data data/clients --output runs` |
| Random/age/no-debt/no-novelty controls and visualization | `cli.run`, `common.plot` | `stair compare ...`; `stair plot --output runs` |

`--input` images are uint8 NHWC (or NHW grayscale); preparation reserves 10% of each client's training samples for private validation and keeps supplied test samples separate. Only private validation probes update recovery/usefulness; test labels are used after training for metrics. `architecture` selects `small_cnn` (the small GroupNorm CNN) or `resnet18` (torchvision ResNet-18 with a 3x3 stride-1 stem and no initial max-pool, no pretrained weights). ResNet batch-normalization buffers remain client-local during pairwise parameter mixing and are included in saved state dictionaries; buffers remain local during communication. A provenance vector tracks per-origin local-update timestamps, and novelty is the fraction of origins newer at the sender. Failed transfers consume attempted model bytes, reset allocation debt, and leave model parameters unchanged. Harmful-exchange rate counts directed successful received updates whose private probe loss rises beyond `loss_tolerance`; the denominator is two per successful pair. P95 runs count missed eligible opportunities since service, pausing when ineligible, and include unfinished runs. Bottom decile averages the worst `ceil(0.1 * clients)` client accuracies.

Allocation controls share input partitions, contacts, local optimization and fixed mixing. `model_byte_budget` is a hard ceiling; use observed common round/byte points for matched comparisons. The native optimizer entry maintains each optimizer’s update and memory state.

## Checks

```bash
python scripts/validate.py
```

Offline checks use generated fixtures to verify method semantics, metrics, gradients and serialization.

## FEMNIST and contact-trace conversion

Standard LEAF shards contain `user_data[writer_id].x` (flattened 28x28 pixels in `[0,1]`) and `.y`. Writer identities remain client groups; conversion keeps writers present in both official train and test shards, stores the exact IDs in `client_ids`, and does not draw a new test split. Subsequent `prepare` reserves validation only from each writer's training samples. Set `clients` to the converter's reported count and `dataset` to `femnist` in your config.

```bash
python scripts/prepare_leaf.py --train path/to/train-shard.json --test path/to/test-shard.json --output data/femnist.npz
stair prepare --config configs/femnist.json --input data/femnist.npz --contacts data/contacts.json --data data/femnist
stair train --config configs/femnist.json --data data/femnist --output runs/femnist
```

For CRAWDAD or another trace, first express its documented native columns as CSV `sender,receiver,start,end` in seconds. The converter bins half-open contact intervals by overlap; IDs are assigned in sorted order unless `--nodes nodes.json` supplies the exact raw node assigned to each client. The generated `.nodes.json` records that mapping. This explicit normalized input avoids guessing a raw trace's column order/time units.

```bash
python scripts/prepare_contacts.py --input data/contact_intervals.csv --output data/contacts.json --clients 20 --rounds 200 --round-seconds 60 --nodes data/nodes.json
```

Set `architecture` to `resnet18` for the larger model. Obtain FEMNIST shards through [LEAF](https://github.com/TalwalkarLab/leaf/tree/master/data/femnist); obtain trace files and their column/time documentation from [CRAWDAD](https://crawdad.org/) or another declared source. Supply their local paths to the converters above.

## External mixing-allocation adapter

An actual external scheduler can export one selected undirected edge list per round, e.g. `[[[0,1],[2,3]],[[0,2]]]`. `--method external --schedule` checks every edge against that round's realized contacts, node range, capacity-one matching and byte budget. It then runs the same local optimizer, fixed symmetric mixer, transport model, private validation/test separation, accuracy/P95/harmful-exchange/byte scorer and checkpoint outputs. A selected external edge need not pass STAIR's own bids: consent/eligibility is a property of the external baseline, while missed eligible opportunities are measured using the same STAIR eligibility signals on its resulting states.

```bash
stair train --config configs/default.json --data data/clients --method external --schedule external_pairings.json --output runs/external_allocator
```

This fixed-mixer adapter is a common allocation experiment. Use the native entry below to execute optimizer-specific update and memory states. The adapter functional check imports fixture pairings, verifies consumed bytes and rejects endpoint collisions.

## Native optimizers and observed matched-byte checkpoints

Native ResNet-18 keeps BatchNorm buffers client-local and saves them with model vectors; SAM updates running statistics once per local step. `python scripts/validate_native_resnet.py` checks gradients and full-state reload. Graph-optimizer iterations can contain multiple neighbor messages; the transport entry below records a separate physical contact clock.

```bash
git submodule update --init --recursive
python -m pip install -e ".[native]"
python scripts/validate_native.py
python -m stair.native --data data/clients --config configs/default.json --methods dpsgd dfedavg matcha dfedsam stair --output runs/native
```

Clone with `--recurse-submodules`. `dpsgd` uses `W x - lr*g(x)`; `dfedavg` runs `local_steps` SGD steps then Metropolis mixing; `dfedsam` computes a SAM perturbation and second backward pass before mixing. `matcha` decomposes contacts into matchings, solves algebraic-connectivity and variance-aware mixing SDPs with CVXPY/Clarabel, then applies sampled activations and the optimized weight. [MATCHA](https://github.com/JYWa/MATCHA) is pinned as a reference; that revision has no standalone license file. CVXPY runs in a separate optimization worker with cached graph designs.

`relaysum` directly calls MIT-licensed [epfml/relaysgd](https://github.com/epfml/relaysgd) `paper-figures/algorithms.py:relaysum_model`, preserving directed message/count memory and initialization correction. Its callback computes training gradients for the selected CNN/ResNet. Canonical execution requires an unchanged connected tree and `success_probability=1`. Use `stair.relay_transport` for deferred delivery over intermittent contacts.

`future_oracle` exactly searches finite-horizon capacity-one matchings with access to future contacts, minimizing lexicographically the largest physical missed-opportunity run then cumulative missed runs. It does not read labels or provide an accuracy bound. This exponential reference is intended for small contact instances. `stair` retains mutual bids/debt/novelty and the fixed local SGD rule. Every method measures missed opportunities using the same state-dependent STAIR eligibility signals, and measures validation harm around the actual mixing phase. Attempted messages consume model bytes even when symmetric transport fails; failures do not count as successful harmful exchanges. A complete native round that exceeds the declared byte budget is rejected.

Each directed model message costs `parameter_count * element_size`; RelaySum count metadata is separate. `comparison.json` records observed final rounds/bytes. `matched_rounds_bytes.json` contains exact common `(round,model_bytes)` points; `matched_final.json` and figures use the last one. With no common point, the matched output is empty.


## Deferred-sweep-v1: executable intermittent RelaySum transport

`stair.relay_transport` implements deferred-sweep-v1 for the pinned `relaysum_model`:

1. A registered connected tree is fixed for the run. Available pending tree edges are served by a lexicographic maximal capacity-one matching; off-tree contacts are ignored. There is no future-contact tree fitting.
2. An epoch computes one original local-gradient/message update. Its message tensors are frozen while physical contacts deliver them. The combined source state is published only after the full tree sweep is delivered. Both **physical round** and **computed/published logical update** clocks are recorded. No equality between those clocks is assumed.
3. Tree and source relay memory remain fixed across temporary disconnections. Undelivered packets retain their epoch and are retried; the next epoch starts only after completion.
4. The selected original core uses world-size normalization with its original initial reference. Failed bilateral deliveries are retried.
5. Every attempt, including failure, charges both actual model vectors against global/per-client model-payload budgets. Two int64 source counts add 16 separately reported counter bytes. Network framing and ACK bytes are not modeled. No over-budget partial vector is sent. An incomplete sweep stays unpublished; checkpoint model vectors and BatchNorm buffers refer to the last published update.

```bash
python scripts/validate_relay_transport.py
python -m stair.relay_transport --data runs/relay-transport-v1/data --config runs/relay-transport-v1/config.json --tree runs/relay-transport-v1/tree.json --output runs/deferred-cli
```

Use the LEAF/CIFAR/contact converters to create `dataset.npz` and `contacts.json`, then supply a registered tree and config. `--success-probability` controls delivery failures. Checkpoints use the native `state`/`buffers` interface. `common_points` reports observed common physical-round/model-byte points and keeps logical update counts visible.
