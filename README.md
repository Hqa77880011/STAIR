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

`--input` images are uint8 NHWC (or NHW grayscale); preparation reserves 10% of each client's training samples for private validation and keeps supplied test samples separate. Only private validation probes update recovery/usefulness; test labels are used after training for metrics. `architecture` selects `small_cnn` (the small GroupNorm CNN) or `resnet18` (torchvision ResNet-18 with a 3x3 stride-1 stem and no initial max-pool, no pretrained weights). ResNet batch-normalization buffers remain client-local during pairwise parameter mixing and are included in saved state dictionaries; this is an explicit normalization choice because the paper does not specify buffer mixing. A provenance vector tracks per-origin local-update timestamps, and novelty is the fraction of origins newer at the sender. Failed transfers consume attempted model bytes, reset allocation debt, and leave model parameters unchanged. Harmful-exchange rate counts directed successful received updates whose private probe loss rises beyond `loss_tolerance`; the denominator is two per successful pair. P95 runs count missed eligible opportunities since service, pausing when ineligible, and include unfinished runs. Bottom decile averages the worst `ceil(0.1 * clients)` client accuracies.

The original allocation-control entry uses the same rounds, input partitions, contacts, local optimizer and fixed mixing rule. `model_byte_budget` is a hard upper bound; unequal observed byte totals are not a matched-byte comparison. CIFAR/NPZ imports and LEAF/contact conversion are supported. The separate native optimizer entry below now implements D-FedAvg/MATCHA and executes the original RelaySum core, retaining their different update/memory states. Functional runs do not reproduce the paper's reported numbers.

## Functional validation

Run the repository-contained check from the repository root after installing dependencies:

```bash
python scripts/validate.py
```

The check uses a fixed seed (17), small generated inputs, the real data preparation and method handlers, hand-computable semantic/metric cases, output serialization and plotting. Generated functional-check artifacts stay in ignored `runs/validation/`; they are not reference experiment results. No full training or benchmark evaluation was performed.

The validation also checks finite cross-entropy, gradients and actual updates of every CNN parameter, two-round client training, and checkpoint reload equality for the same input in `eval()` mode.

Actual validation environment: Windows, Python 3.13.5; numpy 2.1.3, scipy 1.15.3, matplotlib 3.10.0, torch 2.7.1+cu118 (CPU checks), torchvision 0.22.1. On 2026-11-01, `python scripts/validate.py` passed; the distributed implementation and new functional checks do not establish the paper's reported performance.

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

Set `architecture` to `resnet18` in a config to use the larger model. The functional check executes a real ResNet-18 forward/backward/SGD step, checks nonzero finite stem/final-block/classifier gradients, and verifies evaluation output after full state reload. It also reads miniature LEAF-shaped shards and verifies exact contact-bin boundaries. No original FEMNIST or CRAWDAD trace was downloaded or evaluated in this check.

## External mixing-allocation adapter

An actual external scheduler can export one selected undirected edge list per round, e.g. `[[[0,1],[2,3]],[[0,2]]]`. `--method external --schedule` checks every edge against that round's realized contacts, node range, capacity-one matching and byte budget. It then runs the same local optimizer, fixed symmetric mixer, transport model, private validation/test separation, accuracy/P95/harmful-exchange/byte scorer and checkpoint outputs. A selected external edge need not pass STAIR's own bids: consent/eligibility is a property of the external baseline, while missed eligible opportunities are measured using the same STAIR eligibility signals on its resulting states.

```bash
stair train --config configs/default.json --data data/clients --method external --schedule external_pairings.json --output runs/external_allocator
```

This fixed-mixer adapter is a common allocation experiment. Use the native entry below to execute optimizer-specific update and memory states. The adapter functional check imports fixture pairings, verifies consumed bytes and rejects endpoint collisions.

## Native optimizers and observed matched-byte checkpoints

Native ResNet-18 keeps each client's BatchNorm buffers private and includes them with model vectors in checkpoints; SAM updates running statistics once per local step. `python scripts/validate_native_resnet.py` checks real two-client SAM gradients and full-buffer inference parity after reload. Native graph methods may exchange several neighbor messages per optimizer iteration; that iteration is not silently equated with a capacity-one physical contact slot. Their exact intermittent-contact/slot adaptation remains distinct from canonical optimizer execution.

### What the accompanying manuscript does and does not define

The formal STAIR source `AAMAS_2027/main.tex` defines a contact graph per round and at most one bilateral model exchange per client (line 114), matching followed by communication/local update (161, 169–190), common contact/minibatch inputs (232), and equal round/model-byte budgets and reported matched totals (252, 256, 259). It does not define a separate subslot clock or a baseline transport algorithm. The capacity-one premise explicitly concerns half-duplex/energy-limited bilateral exchange (277).

The original RelaySum code is available and includes more than the selected academic core: [RelayMechanism](https://github.com/epfml/relaysgd/blob/e1e332bfaf6a2b29f1677b7fa31a27b6e0d2802a/utils/communication.py) implements `message_drop_prob`, `overlap`, `world_size`/`counts` normalization and an updateable reference state; the [official robustness experiment](https://github.com/epfml/relaysgd/blob/e1e332bfaf6a2b29f1677b7fa31a27b6e0d2802a/experiments/neurips21-cifar-robustness.py) selects model relaying, world-size normalization and reference-state updates. The [original paper, Section 5.4](https://papers.nips.cc/paper/2021/file/ebbdfea212e3a756a1fded7b35578525-Paper.pdf), supplies a robust missing-model update. Thus the remaining issue is not missing public code or weights. Those fixed-topology/dropped-message facilities do not specify the following STAIR experiment rules:

| Missing protocol field | Why existing inputs and source do not determine it |
|---|---|
| Baseline contact arbitration: `E_t → M_t` | `contacts.json` records available edges, not the capacity-one matching selected by RelaySum or the other graph baselines. The manuscript gives STAIR's bid/mutual-max rule but no baseline matching policy, tie order, or rule for applying mutual eligibility to a baseline. MATCHA's independently active matching union can also exceed degree one. |
| Physical-round to relay-iteration mapping and local-update clock | A three-worker relay tree has a degree-two center and sends along both tree edges per source iteration. Serving these edges in two capacity-one rounds requires deciding whether SGD runs once or twice, whether messages use one frozen snapshot, and whether a pending sweep can span an expired contact. Lines 161/190/252 do not supply this mapping. |
| Relay tree and message epoch/cache policy under disconnection | The edge-list trace does not specify a fixed union tree, a per-component/per-round tree, root/rebuild rules, or whether messages/counts from inactive/previous-tree edges survive. Rebuilding a tree can change which workers a cached subtree sum represents; counts alone do not identify overlapping origins. |
| Selected loss/normalization/reference rule | The source offers concrete alternatives, including an original robust rule. The STAIR config's scalar `success_probability` does not select these options or state whether a missing contact is an unattempted exchange, a delivered-then-discarded simulated message, or a deferred relay message. This changes both states and traffic. |
| Exact shared model-byte schedule and endpoint charging | `model_byte_budget` is a global ceiling, not an exact per-round transfer quota or a specified final target with a catch-up/idle rule. The paper also mentions endpoint affordability without supplying per-client budgets. Equal ceilings and round counts do not force equal actual payload bytes. A failure's attempted/transmitted/received-byte charging and incomplete-sweep treatment are not specified. |

`prepare_contacts.py` makes trace bin width, phase and client mapping explicit inputs. Those are also needed to reproduce the original trace, but even a fully supplied contact edge list does not supply the five rules above. `matched_rounds_bytes.json` honestly finds observed common points; it is not a reconstruction of an unspecified shared allocation schedule.

```bash
python scripts/check_relay_protocol.py
```

This diagnostic executes the unchanged academic RelaySum core with a deliberately naive dynamic-neighbor substitution. With three zero-initialized workers, a single first-round update and contacts `[(0,1)]`, then `[(1,2)]`, its states are `[1,1,0]` and `[1/3,1/3,1/3]`: worker 0 changes in round two despite no gradient or received message. Freezing an unmatched model, retaining old buffers, or choosing the original robust reference rule would give different behavior. A second check shows that, at one common physical round and a common large byte ceiling, default STAIR can select no eligible edge (0 payload bytes), while a two-node full relay iteration sends two scalar float32 vectors (8 bytes). Neither example is a benchmark result or an implemented intermittent baseline.

The manuscript therefore does not sufficiently define the reported unified intermittent RelaySum/capacity-one/same-round/same-byte protocol. The canonical native entry remains executable and the missing-definition boundary remains explicit. No invented tree, sweep clock, cache policy or transfer quota is presented as the paper's original algorithm.

```bash
git submodule update --init --recursive
python -m pip install -e ".[native]"
python scripts/validate_native.py
python -m stair.native --data data/clients --config configs/default.json --methods dpsgd dfedavg matcha dfedsam stair --output runs/native
```

Clone with `--recurse-submodules`. `dpsgd` implements `W x - lr*g(x)`; `dfedavg` performs `local_steps` local SGD updates then Metropolis neighbor mixing; `dfedsam` computes the actual SAM perturbation and second backward pass before neighbor mixing. `matcha` decomposes each contact graph into matchings and solves the published algebraic-connectivity probability SDP and variance-aware mixing-weight SDP using CVXPY/Clarabel. Its Bernoulli matching activations and optimized weight are actually applied. This is a native implementation of [MATCHA](https://github.com/JYWa/MATCHA), not a claim that its MPI training script was run. The pinned upstream is retained as reference; that checkout has no standalone license file. CVXPY runs in a pure optimization worker so the Anaconda/MKL and PyTorch OpenMP DLLs do not collide on Windows. Graph designs are cached and no PyTorch import occurs in this worker.

`relaysum` executes `relaysum_model` directly from the fixed MIT [epfml/relaysgd](https://github.com/epfml/relaysgd) source, including directed relay-message/count memory and its initialization correction. Its gradient callback runs the actual selected CNN/ResNet on each client's training minibatch. This entry selects the authors' academic `paper-figures/algorithms.py:relaysum_model`, whose interface has no loss/overlap policy. The current runner therefore restricts this selected core to an unchanged connected tree with reliable transport and rejects unsupported dynamic/disconnected/loss inputs. This restriction does not describe every original RelaySum implementation: the official distributed core includes dropped-message robustness, overlap and normalization options, as detailed below. It does not label ordinary pair averaging as RelaySum. Use `--methods ... relaysum` only with a static-tree dataset/config (`success_probability=1`). The independent deferred-sweep-v1 transport below now executes this original core under intermittent physical contacts. The paper's exact historical transport configuration remains unverified.

`future_oracle` exactly searches finite-horizon capacity-one matchings with access to future contacts, minimizing lexicographically the largest physical missed-opportunity run then cumulative missed runs. It does not read labels or provide an accuracy bound. This exponential reference is intended for small contact instances. `stair` retains mutual bids/debt/novelty and the fixed local SGD rule. Every method measures missed opportunities using the same state-dependent STAIR eligibility signals, and measures validation harm around the actual mixing phase. Attempted messages consume model bytes even when symmetric transport fails; failures do not count as successful harmful exchanges. A complete native round that exceeds the declared byte budget is rejected.

Each real directed model message contributes `parameter_count * element_size`; RelaySum count metadata is recorded separately. `comparison.json` records each method's final **observed** rounds/bytes. `matched_rounds_bytes.json` contains only exact common `(round,model_bytes)` points across the requested methods; `matched_final.json`/figures use the last such point. No interpolation, rescaling or forced message truncation manufactures a match. If no joint point exists, the matched output is empty. The CPU check runs all five native baselines, STAIR and the future-contact oracle with genuine CNN gradients, reloads saved vectors, hand-checks the two-node MATCHA design (`p=1, alpha≈0.5`), tests RelaySum's nontrivial three-node relay memory, checks exact common points and rejects budget overflow/dynamic canonical RelaySum. The small fixture uses an explicit initial age of one to create an initially eligible exchange. Full FEMNIST/CIFAR/contact-trace benchmark training and the paper's complete same-round/same-byte intermittent protocol were not run.


## Deferred-sweep-v1: executable intermittent RelaySum transport

`stair.relay_transport` now supplies a **new, explicit** transport protocol for the unchanged pinned `relaysum_model`. It uses actual source-produced directed model tensors and counts, not pair averaging or imported optimizer outputs. It resolves an executable public choice for each missing field without calling it the original manuscript protocol:

1. A registered connected tree is fixed for the run. Available pending tree edges are served by a lexicographic maximal capacity-one matching; off-tree contacts are ignored. There is no future-contact tree fitting.
2. An epoch computes one original local-gradient/message update. Its message tensors are frozen while physical contacts deliver them. The combined source state is published only after the full tree sweep is delivered. Both **physical round** and **computed/published logical update** clocks are recorded. No equality between those clocks is assumed.
3. Tree and source relay memory remain fixed across temporary disconnections. Undelivered packets retain their epoch and are retried; the next epoch starts only after completion.
4. The selected original core uses world-size normalization with its original initial reference. Loss means a failed bilateral delivery attempt followed by retransmission; it is not the distributed core's alternative reference/overlap/drop-normalization algorithm.
5. Every attempt, including failure, charges both actual model vectors against global/per-client model-payload budgets. Two int64 source counts add 16 separately reported counter bytes. Network framing and ACK bytes are not modeled. No over-budget partial vector is sent. An incomplete sweep stays unpublished; checkpoint model vectors and BatchNorm buffers refer to the last published update.

```bash
python scripts/validate_relay_transport.py
python -m stair.relay_transport --data runs/relay-transport-v1/data --config runs/relay-transport-v1/config.json --tree runs/relay-transport-v1/tree.json --output runs/deferred-cli
```

For real input, use the existing LEAF/CIFAR/contact converters to create `dataset.npz` and `contacts.json`, and supply a separately registered tree and config. `--success-probability` controls failed delivery attempts. The saved checkpoint uses the same `state`/`buffers` interface as native optimizer checkpoints. `common_points` returns only actually observed common physical-round/model-byte points; it never interpolates or forces a byte match. Logical update counts remain visible at every point, so the resulting protocol must not be described as identical-SGD-clock reproduction of the paper.

Actual CPU1 checks passed: a three-node tree's four intermittent slots complete two original RelaySum updates, exactly matching the unchanged source's two full iterations with nonzero real autograd gradients. Model/count payloads were 64/64 bytes, with at most one exchange per client per slot. Probability-zero loss consumed attempted bytes while publishing no false state; under-budget and interrupted-epoch cases remained explicit. A real two-client CNN also performed two nonzero gradient updates and matched accuracy after complete model/BatchNorm reload. Generated fixtures validate this transport and are not FEMNIST/CRAWDAD benchmark results. The original private clock/tree/quota settings and historical manuscript numbers still require the original experiment artifacts.
