import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.models import resnet18
from .common import load, save, table, plot
from .core import allocation


class Network(nn.Module):
    def __init__(self, channels, classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, classes),
        )

    def forward(self, x):
        return self.layers(x)


def build_model(channels, classes, architecture="small_cnn"):
    """Build the small CNN or an unpretrained CIFAR/FEMNIST ResNet-18."""
    if architecture == "small_cnn":
        return Network(channels, classes)
    if architecture == "resnet18":
        model = resnet18(weights=None, num_classes=classes)
        model.conv1 = nn.Conv2d(channels, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model
    raise ValueError("Unknown model architecture: " + architecture)


def imported_matching(round_edges, contacts, clients):
    """Validate external capacity-one pairings against realized physical contacts."""
    selected = []
    used = set()
    physical = {tuple(sorted(edge)) for edge in contacts}
    for pair in round_edges:
        if len(pair) != 2 or any(
            not isinstance(i, int) or not 0 <= i < clients for i in pair
        ):
            raise ValueError("External pairings require two valid client indices")
        edge = tuple(sorted(pair))
        if edge[0] == edge[1] or edge not in physical or set(edge) & used:
            raise ValueError(
                "External schedule must be a physical capacity-one matching"
            )
        selected.append(edge)
        used.update(edge)
    return selected


def prepare(a, c):
    root = Path(a.data)
    root.mkdir(parents=True, exist_ok=True)
    if a.input:
        data = np.load(a.input)
        x, y, client = data["x"], data["y"], data["client"]
        test_x, test_y, test_client = (
            data["test_x"],
            data["test_y"],
            data["test_client"],
        )
    else:
        if c["dataset"] not in ("cifar10", "cifar100"):
            raise ValueError("Use --input with converted NPZ for non-CIFAR datasets")
        dataset = CIFAR10 if c["dataset"] == "cifar10" else CIFAR100
        tr = dataset(root / "raw", train=True, download=True)
        te = dataset(root / "raw", train=False, download=True)
        x, y = np.array(tr.data), np.array(tr.targets)
        test_x, test_y = np.array(te.data), np.array(te.targets)
        rng = np.random.default_rng(c["seed"])
        client = np.empty(len(y), int)
        test_client = np.empty(len(test_y), int)
        for label in np.unique(y):
            distribution = rng.dirichlet(np.full(c["clients"], c["alpha"]))
            for yy, cc in [(y, client), (test_y, test_client)]:
                indices = np.flatnonzero(yy == label)
                rng.shuffle(indices)
                sizes = rng.multinomial(len(indices), distribution)
                for i, part in enumerate(np.split(indices, np.cumsum(sizes)[:-1])):
                    cc[part] = i
    split = np.full(len(y), "train", dtype="<U5")
    rng = np.random.default_rng(c["seed"])
    for i in np.unique(client):
        ids = np.flatnonzero(client == i)
        rng.shuffle(ids)
        if len(ids) < 4:
            raise ValueError("At least four training samples per client required")
        split[ids[: max(1, len(ids) // 10)]] = "val"
    np.savez_compressed(
        root / "dataset.npz",
        x=x,
        y=y,
        client=client,
        split=split,
        test_x=test_x,
        test_y=test_y,
        test_client=test_client,
    )
    if a.contacts:
        contacts = load(a.contacts)
    else:
        contacts = []
        bridge = True
        n = c["clients"]
        half = n // 2
        for t in range(c["rounds"]):
            if rng.random() < c["bridge_failure"]:
                bridge = not bridge
            edges = [
                (i, j)
                for i in range(n)
                for j in range(i + 1, n)
                if i // half == j // half and rng.random() < 0.35
            ]
            if bridge:
                edges.extend(
                    (i, half) for i in range(min(c["bridge_competitors"], half))
                )
            contacts.append(sorted(set(edges)))
    save(root / "contacts.json", contacts)


def run(a, c, method):
    torch.manual_seed(c["seed"])
    rng = np.random.default_rng(c["seed"])
    data = np.load(Path(a.data) / "dataset.npz")
    contacts = load(Path(a.data) / "contacts.json")
    external = None
    if method == "external":
        if not getattr(a, "schedule", None):
            raise ValueError(
                "External mixing evaluation requires --schedule pairings.json"
            )
        external = load(a.schedule)
        if len(external) < min(c["rounds"], len(contacts)):
            raise ValueError("External schedule omits an evaluated contact round")
    device = c["device"]
    n = c["clients"]

    def images(key):
        x = torch.from_numpy(data[key].copy()).float()
        if x.ndim == 3:
            x = x.unsqueeze(-1)
        return x.permute(0, 3, 1, 2) / 255

    x = images("x")
    test = images("test_x")
    y = torch.from_numpy(data["y"].astype("int64"))
    test_y = data["test_y"]
    prototype = build_model(
        x.shape[1],
        c.get("classes", int(y.max()) + 1),
        c.get("architecture", "small_cnn"),
    ).to(device)
    models = [deepcopy(prototype) for _ in range(n)]
    fit = []
    probes = []
    tests = []
    for i in range(n):
        fit.append(np.flatnonzero((data["client"] == i) & (data["split"] == "train")))
        probes.append(np.flatnonzero((data["client"] == i) & (data["split"] == "val")))
        tests.append(np.flatnonzero(data["test_client"] == i))
        if not len(fit[-1]) or not len(probes[-1]) or not len(tests[-1]):
            raise ValueError("Every client needs training, validation and test samples")

    def loss(i, model):
        model.eval()
        with torch.no_grad():
            return float(
                nn.functional.cross_entropy(
                    model(x[probes[i]].to(device)), y[probes[i]].to(device)
                )
            )

    optimizers = [torch.optim.SGD(m.parameters(), lr=c["lr"]) for m in models]
    h = np.full((n, n), c["usefulness_prior"])
    age = np.zeros(n)
    debt = np.zeros(n)
    provenance = np.zeros((n, n), int)
    np.fill_diagonal(provenance, 1)
    references = np.array([loss(i, m) for i, m in enumerate(models)])
    losses_history = [[] for _ in range(n)]
    current_run = np.zeros(n, int)
    runs = []
    history = []
    trace = []
    payload = 2 * parameters_to_vector(prototype.parameters()).numel() * 4
    total_bytes = harmful = exchanges = 0
    for t in range(min(c["rounds"], len(contacts))):
        before = np.array([loss(i, m) for i, m in enumerate(models)])
        sigma = np.array([max(np.std(v), 0.01) if v else 0.01 for v in losses_history])
        recovery = np.clip((before - references) / (sigma + 1e-8), 0, 10)
        edges = [tuple(sorted(map(int, e))) for e in contacts[t]]
        selected_override = (
            imported_matching(external[t], edges, n) if external is not None else None
        )
        selected, eligible, debt, novelty, bids = allocation(
            edges,
            recovery,
            age,
            debt,
            h,
            provenance,
            c,
            "stair" if method == "external" else method,
            rng,
            selected_override=selected_override,
        )
        remaining = max(0, (c["model_byte_budget"] - total_bytes) // payload)
        if len(selected) > remaining:
            raise ValueError(
                "Byte budget exhausted before a complete allocated round; reduce rounds for all methods"
            )
        served = {v for e in selected for v in e}
        refreshed = set()
        round_events = []
        for i in range(n):
            if i in served:
                if current_run[i]:
                    runs.append(int(current_run[i]))
                current_run[i] = 0
            elif i in eligible:
                current_run[i] += 1
        for i, j in selected:
            total_bytes += payload
            ok = rng.random() < c["success_probability"]
            exchanges += 1
            if ok:
                wi = parameters_to_vector(models[i].parameters()).detach().clone()
                wj = parameters_to_vector(models[j].parameters()).detach().clone()
                alpha = c["mixing"]
                if not 0 < alpha <= 0.5:
                    raise ValueError("mixing must lie in (0,0.5]")
                vector_to_parameters(
                    (1 - alpha) * wi + alpha * wj, models[i].parameters()
                )
                vector_to_parameters(
                    alpha * wi + (1 - alpha) * wj, models[j].parameters()
                )
                after_i, after_j = loss(i, models[i]), loss(j, models[j])
                harmful += int(after_i > before[i] + c["loss_tolerance"]) + int(
                    after_j > before[j] + c["loss_tolerance"]
                )
                for receiver, sender, after in [(i, j, after_i), (j, i, after_j)]:
                    if novelty[receiver, sender] >= c["novelty_min"]:
                        gain = np.clip(
                            (before[receiver] - after) / (sigma[receiver] + 1e-8), -1, 1
                        )
                        h[receiver, sender] = (1 - c["eta"]) * h[receiver, sender] + c[
                            "eta"
                        ] * max(gain, 0)
                        refreshed.add(receiver)
                union = np.maximum(provenance[i], provenance[j])
                provenance[i] = union
                provenance[j] = union
            round_events.append(dict(edge=[i, j], success=ok))
        for i in range(n):
            age[i] = 0 if i in refreshed else age[i] + 1
            models[i].train()
            for _ in range(c["local_epochs"]):
                indices = rng.permutation(fit[i])
                for start in range(0, len(indices), c["batch_size"]):
                    ids = indices[start : start + c["batch_size"]]
                    objective = nn.functional.cross_entropy(
                        models[i](x[ids].to(device)), y[ids].to(device)
                    )
                    optimizers[i].zero_grad()
                    objective.backward()
                    optimizers[i].step()
            provenance[i, i] = t + 2
            new_loss = loss(i, models[i])
            losses_history[i].append(new_loss)
            losses_history[i] = losses_history[i][-20:]
            references[i] = np.median(losses_history[i])
        history.append(
            dict(
                round=t + 1,
                model_bytes=total_bytes,
                mean_debt=debt.mean(),
                mean_age=age.mean(),
                exchanges=len(selected),
                validation_loss=np.mean([loss(i, m) for i, m in enumerate(models)]),
            )
        )
        trace.append(
            dict(
                round=t + 1,
                eligible=sorted(eligible),
                selected=round_events,
                debt=debt.copy(),
                age=age.copy(),
            )
        )
    runs.extend(current_run[current_run > 0].tolist())
    accuracy = []
    for i, m in enumerate(models):
        m.eval()
        correct = 0
        with torch.no_grad():
            for start in range(0, len(tests[i]), c["batch_size"]):
                ids = tests[i][start : start + c["batch_size"]]
                correct += np.sum(
                    m(test[ids].to(device)).argmax(1).cpu().numpy() == test_y[ids]
                )
        accuracy.append(correct / len(tests[i]))
    summary = dict(
        method=method,
        mean_accuracy=np.mean(accuracy),
        bottom_decile_accuracy=np.mean(
            np.sort(accuracy)[: max(1, int(np.ceil(0.1 * n)))]
        ),
        p95_missed_eligible_run=np.quantile(runs, 0.95) if runs else 0.0,
        harmful_exchange_rate=harmful
        / max(1, 2 * sum(e["success"] for row in trace for e in row["selected"])),
        model_bytes=total_bytes,
    )
    output = Path(a.output) / method
    output.mkdir(parents=True, exist_ok=True)
    save(output / "metrics.json", summary)
    save(output / "trace.json", trace)
    save(output / "client_accuracy.json", accuracy)
    table(output / "history.csv", history)
    save(output / "config.json", c)
    torch.save([m.cpu().state_dict() for m in models], output / "models.pt")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["prepare", "train", "compare", "plot"])
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--input")
    p.add_argument("--contacts")
    p.add_argument("--data", default="data/clients")
    p.add_argument("--output", default="runs")
    p.add_argument(
        "--schedule", help="External per-round capacity-one mixing edge lists JSON"
    )
    p.add_argument(
        "--method",
        choices=["stair", "random", "age", "no_debt", "no_novelty", "external"],
        default="stair",
    )
    a = p.parse_args()
    c = load(a.config)
    if a.command == "prepare":
        prepare(a, c)
    elif a.command == "train":
        run(a, c, a.method)
    elif a.command == "compare":
        rows = [
            run(a, c, m) for m in ["random", "age", "no_debt", "no_novelty", "stair"]
        ]
        save(Path(a.output) / "comparison.json", rows)
        table(Path(a.output) / "comparison.csv", rows)
    else:
        plot(Path(a.output) / "comparison.json", Path(a.output) / "figures")
