"""Native decentralized optimizers and exact observed-byte comparison points."""

import argparse
from functools import lru_cache
import importlib.util
from pathlib import Path
import sys

import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .cli import build_model
from .common import load, save, table, plot
from .core import allocation


def future_contact_oracle(contacts, clients):
    """Exact future-contact matching oracle for lexicographic physical service tails.

    Minimize (largest missed run, cumulative missed runs), without accessing labels.
    This is an upper reference for the specified service objective, not accuracy.
    """
    options = []
    for edges in contacts:
        possible = []

        def extend(index, chosen, used):
            if index == len(edges):
                if all(i in used or j in used for i, j in edges):
                    possible.append(tuple(chosen))
                return
            i, j = edges[index]
            extend(index + 1, chosen, used)
            if i not in used and j not in used:
                extend(index + 1, chosen + [(i, j)], used | {i, j})

        extend(0, [], set())
        options.append(possible)

    @lru_cache(None)
    def solve(t, missed, largest):
        if t == len(contacts):
            return (largest, 0), ()
        available = {i for edge in contacts[t] for i in edge}
        best = None
        for matching in options[t]:
            served = {i for edge in matching for i in edge}
            current = tuple(
                0 if i in served else missed[i] + int(i in available)
                for i in range(clients)
            )
            objective, suffix = solve(t + 1, current, max(largest, max(current)))
            candidate = (
                (objective[0], objective[1] + sum(current)),
                (matching,) + suffix,
            )
            if best is None or candidate < best:
                best = candidate
        return best

    objective, schedule = solve(0, (0,) * clients, 0)
    return dict(
        schedule=schedule,
        max_missed_run=objective[0],
        cumulative_missed_runs=objective[1],
    )


def matcha_design(edges, clients, budget=0.5):
    return cached_matcha_design(tuple(tuple(edge) for edge in edges), clients, budget)


@lru_cache(None)
def cached_matcha_design(edges, clients, budget):
    """Solve the exact MATCHA SDPs in a worker without importing PyTorch there."""
    import json
    import subprocess

    worker = Path(__file__).with_name("matcha_solver.py")
    result = subprocess.run(
        [sys.executable, str(worker)],
        input=json.dumps(dict(edges=edges, clients=clients, budget=budget)),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError("MATCHA SDP worker failed: " + result.stderr + result.stdout)
    return json.loads(result.stdout.splitlines()[-1])


def metropolis(edges, clients):
    """Construct a symmetric, average-preserving neighbor mixing matrix."""
    degree = np.zeros(clients, int)
    for i, j in edges:
        degree[i] += 1
        degree[j] += 1
    weights = np.eye(clients)
    for i, j in edges:
        value = 1 / (1 + max(degree[i], degree[j]))
        weights[i, j] = weights[j, i] = value
        weights[i, i] -= value
        weights[j, j] -= value
    return torch.tensor(weights, dtype=torch.float32)


def relay_core():
    """Load the authors' unmodified single-process RelaySum implementation."""
    path = Path(__file__).resolve().parents[2] / "upstream/relaysgd/paper-figures"
    if not (path / "algorithms.py").is_file():
        raise FileNotFoundError("Initialize the pinned relaysgd submodule")
    sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location(
        "stair_upstream_relay", path / "algorithms.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.relaysum_model


class TreeWorld:
    """The static tree topology required by the canonical RelaySum implementation."""

    def __init__(self, clients, edges):
        self.num_workers = clients
        self.workers = range(clients)
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.workers)
        self.graph.add_edges_from(edges)
        if not nx.is_tree(self.graph):
            raise ValueError("Canonical RelaySum requires a connected tree")
        self.max_degree = max(dict(self.graph.degree()).values())

    def neighbors(self, worker):
        return list(self.graph.neighbors(worker))


class LearningTask:
    """Actual per-client minibatch gradients, independent of evaluation labels."""

    def __init__(self, data, config):
        self.config, self.calls, self.grad_norms = config, 0, []
        self.rng = np.random.default_rng(config["seed"])
        self.x = self.images(data["x"])
        self.test = self.images(data["test_x"])
        self.y = torch.tensor(data["y"], dtype=torch.long)
        self.test_y = torch.tensor(data["test_y"], dtype=torch.long)
        self.train = [
            np.flatnonzero((data["client"] == i) & (data["split"] == "train"))
            for i in range(config["clients"])
        ]
        self.val = [
            np.flatnonzero((data["client"] == i) & (data["split"] == "val"))
            for i in range(config["clients"])
        ]
        self.test_ids = [
            np.flatnonzero(data["test_client"] == i) for i in range(config["clients"])
        ]
        if any(
            not len(ids)
            for split in (self.train, self.val, self.test_ids)
            for ids in split
        ):
            raise ValueError("Every client requires train/validation/test samples")
        torch.manual_seed(config["seed"])
        self.model = build_model(
            self.x.shape[1],
            config.get("classes", int(self.y.max()) + 1),
            config.get("architecture", "small_cnn"),
        )
        initial = parameters_to_vector(self.model.parameters()).detach().clone()
        self.initial = initial.repeat(config["clients"], 1).unsqueeze(-1)
        self.buffers = [
            {name: value.detach().clone() for name, value in self.model.named_buffers()}
            for _ in range(config["clients"])
        ]

    def use_buffers(self, index):
        for name, value in self.model.named_buffers():
            value.copy_(self.buffers[index][name])

    def restore(self, checkpoint):
        """Restore client-local normalization state and return model vectors."""
        self.buffers = checkpoint["buffers"]
        return checkpoint["state"]

    @staticmethod
    def images(array):
        value = torch.from_numpy(array.copy()).float()
        if value.ndim == 3:
            value = value.unsqueeze(-1)
        return value.permute(0, 3, 1, 2) / 255

    def init_state(self):
        return self.initial.clone()

    def gradient(self, state, sam=False):
        values = []
        for i, vector in enumerate(state):
            self.use_buffers(i)
            vector_to_parameters(vector.flatten().clone(), self.model.parameters())
            ids = self.rng.choice(
                self.train[i],
                min(len(self.train[i]), self.config["batch_size"]),
                replace=False,
            )
            self.model.train()
            self.model.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(self.model(self.x[ids]), self.y[ids]).backward()
            first = (
                parameters_to_vector([p.grad for p in self.model.parameters()])
                .detach()
                .clone()
            )
            if sam:
                first_buffers = {
                    name: value.clone() for name, value in self.model.named_buffers()
                }
                perturbation = (
                    self.config.get("sam_rho", 0.05) * first / (first.norm() + 1e-12)
                )
                vector_to_parameters(
                    vector.flatten() + perturbation, self.model.parameters()
                )
                self.model.zero_grad(set_to_none=True)
                nn.functional.cross_entropy(
                    self.model(self.x[ids]), self.y[ids]
                ).backward()
                first = (
                    parameters_to_vector([p.grad for p in self.model.parameters()])
                    .detach()
                    .clone()
                )
                for name, value in self.model.named_buffers():
                    value.copy_(first_buffers[name])
            self.buffers[i] = {
                name: value.detach().clone()
                for name, value in self.model.named_buffers()
            }
            values.append(first.unsqueeze(-1))
        result = torch.stack(values)
        self.calls += 1
        self.grad_norms.append(float(result.norm()))
        return result

    def grad(self, state):
        result = self.gradient(state)
        self.last_local_state = state - self.config["lr"] * result
        return result

    def validation(self, state):
        values = []
        self.model.eval()
        with torch.no_grad():
            for i, vector in enumerate(state):
                self.use_buffers(i)
                vector_to_parameters(vector.flatten().clone(), self.model.parameters())
                values.append(
                    float(
                        nn.functional.cross_entropy(
                            self.model(self.x[self.val[i]]), self.y[self.val[i]]
                        )
                    )
                )
        return np.array(values)

    def accuracy(self, state):
        values = []
        self.model.eval()
        with torch.no_grad():
            for i, vector in enumerate(state):
                self.use_buffers(i)
                vector_to_parameters(vector.flatten().clone(), self.model.parameters())
                ids = self.test_ids[i]
                values.append(
                    float(
                        (self.model(self.test[ids]).argmax(1) == self.test_y[ids])
                        .float()
                        .mean()
                    )
                )
        return values


def run(data, contacts, config, method, output):
    """Run a named optimizer and record actual model traffic at every round."""
    n = config["clients"]
    task = LearningTask(data, config)
    state = task.init_state()
    rounds = min(config["rounds"], len(contacts))
    edge_sets = [
        sorted(set(tuple(sorted(e)) for e in edges)) for edges in contacts[:rounds]
    ]
    if any(
        i == j or not 0 <= i < n or not 0 <= j < n
        for edges in edge_sets
        for i, j in edges
    ):
        raise ValueError("Contacts contain an invalid client edge")
    relay = None
    oracle = future_contact_oracle(edge_sets, n) if method == "future_oracle" else None
    if method == "relaysum":
        if config["success_probability"] != 1.0:
            raise ValueError("Canonical RelaySum requires reliable tree links")
        if any(edges != edge_sets[0] for edges in edge_sets):
            raise ValueError(
                "Canonical upstream RelaySum needs a static tree; no dynamic-contact result is invented"
            )
        relay = relay_core()(task, TreeWorld(n, edge_sets[0]), config["lr"], rounds)
        next(relay)
    rng = np.random.default_rng(config["seed"])
    designs = {}
    model_bytes = metadata_bytes = harmful = endpoints = 0
    missed = np.zeros(n, int)
    age = np.full(n, config.get("initial_age", 0.0))
    debt = np.zeros(n)
    usefulness = np.full((n, n), config["usefulness_prior"])
    provenance = np.eye(n, dtype=int)
    references = task.validation(state)
    loss_history = [[] for _ in range(n)]
    runs, trace, snapshots = [], [], []
    parameter_bytes = state[0].numel() * state.element_size()
    for t, edges in enumerate(edge_sets):
        active = edges
        alpha = None
        before = task.validation(state)
        sigma = np.array([max(np.std(v), 0.01) if v else 0.01 for v in loss_history])
        recovery = np.clip((before - references) / (sigma + 1e-8), 0, 10)
        if method == "stair":
            active, _, _, _, _ = allocation(
                edges, recovery, age, debt, usefulness, provenance, config, "stair", rng
            )
        if oracle is not None:
            active = list(oracle["schedule"][t])
        if method == "matcha":
            key = tuple(edges)
            if key not in designs:
                designs[key] = matcha_design(edges, n, config.get("matcha_budget", 0.5))
            design = designs[key]
            active = [
                tuple(edge)
                for group, p in zip(design["matchings"], design["probability"])
                if rng.random() < p
                for edge in group
            ]
            alpha = design["alpha"]
        _, available, debt, novelty, _ = allocation(
            edges,
            recovery,
            age,
            debt,
            usefulness,
            provenance,
            config,
            "stair",
            rng,
            selected_override=active,
        )
        if (
            model_bytes + 2 * len(active) * parameter_bytes
            > config["model_byte_budget"]
        ):
            raise ValueError(
                "Model-byte budget cannot fund the next complete native round"
            )
        attempted = active
        active = [
            edge for edge in attempted if rng.random() < config["success_probability"]
        ]
        weights = metropolis(active, n)
        if alpha is not None:
            weights = torch.eye(n)
            for i, j in active:
                weights[i, i] -= alpha
                weights[j, j] -= alpha
                weights[i, j] += alpha
                weights[j, i] += alpha
        if relay is not None:
            # The upstream algorithm includes relay-message/count memory and its own SGD.
            state = next(relay).state.clone()
            mixed = state
            before = task.validation(task.last_local_state)
            metadata_bytes += 2 * len(active) * 8
        elif method in ("dpsgd", "stair", "future_oracle"):
            mixed = torch.einsum("ij,jdk->idk", weights, state)
            state = mixed - config["lr"] * task.gradient(
                state if method == "dpsgd" else mixed
            )
        else:
            for _ in range(config.get("local_steps", 1)):
                state = state - config["lr"] * task.gradient(
                    state, sam=method == "dfedsam"
                )
            mixed = torch.einsum("ij,jdk->idk", weights, state)
            # Evaluate mixing harm against the actual locally updated state.
            before = task.validation(state)
            state = mixed
        after = task.validation(mixed)
        served = {i for edge in attempted for i in edge}
        refreshed = set()
        old_provenance = provenance.copy()
        for i, j in active:
            for receiver, sender in ((i, j), (j, i)):
                if novelty[receiver, sender] >= config["novelty_min"]:
                    gain = np.clip(
                        (before[receiver] - after[receiver]) / (sigma[receiver] + 1e-8),
                        -1,
                        1,
                    )
                    usefulness[receiver, sender] = (1 - config["eta"]) * usefulness[
                        receiver, sender
                    ] + config["eta"] * max(gain, 0)
                    refreshed.add(receiver)
                provenance[receiver] = np.maximum(
                    provenance[receiver], old_provenance[sender]
                )
        for i in range(n):
            if i in served:
                if missed[i]:
                    runs.append(int(missed[i]))
                missed[i] = 0
            elif i in available:
                missed[i] += 1
            age[i] = 0 if i in refreshed else age[i] + 1
            provenance[i, i] = t + 2
        final_loss = task.validation(state)
        for i in range(n):
            loss_history[i] = (loss_history[i] + [final_loss[i]])[-20:]
            references[i] = np.median(loss_history[i])
        model_bytes += 2 * len(attempted) * parameter_bytes
        harmful += sum(
            after[i] > before[i] + config["loss_tolerance"]
            for edge in active
            for i in edge
        )
        endpoints += 2 * len(active)
        accuracy = task.accuracy(state)
        snapshot = dict(
            method=method,
            round=t + 1,
            model_bytes=model_bytes,
            metadata_bytes=metadata_bytes,
            mean_accuracy=float(np.mean(accuracy)),
            bottom_decile_accuracy=float(
                np.mean(sorted(accuracy)[: max(1, int(np.ceil(n / 10)))])
            ),
            harmful_exchange_rate=harmful / max(1, endpoints),
            p95_missed_eligible_run=float(
                np.quantile(runs + missed[missed > 0].tolist(), 0.95)
            )
            if runs or missed.any()
            else 0.0,
            gradient_calls=task.calls,
        )
        snapshots.append(snapshot)
        trace.append(
            dict(
                round=t + 1,
                edges=active,
                attempted_edges=attempted,
                mixing_matrix=weights.numpy(),
                eligible=sorted(available),
                debt=debt,
                age=age,
                validation_before=before,
                validation_after_mixing=after,
            )
        )
    output = Path(output) / method
    save(output / "history.json", snapshots)
    save(output / "trace.json", trace)
    save(output / "matcha_designs.json", list(designs.values()))
    if oracle is not None:
        save(output / "future_contact_oracle.json", oracle)
    save(output / "metrics.json", snapshots[-1])
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(state=state, buffers=task.buffers, config=config, method=method),
        output / "models.pt",
    )
    return snapshots, task


def compare(data, contacts, config, output, methods=None):
    """Only label comparisons matched when both observed rounds and bytes agree."""
    methods = methods or [
        "dpsgd",
        "dfedavg",
        "matcha",
        "relaysum",
        "dfedsam",
        "stair",
        "future_oracle",
    ]
    histories = [run(data, contacts, config, method, output)[0] for method in methods]
    indexes = [
        {(row["round"], row["model_bytes"]): row for row in rows} for rows in histories
    ]
    common = set.intersection(*(set(index) for index in indexes))
    matched = [index[key] for key in sorted(common) for index in indexes]
    save(Path(output) / "matched_rounds_bytes.json", matched)
    save(Path(output) / "comparison.json", [rows[-1] for rows in histories])
    table(Path(output) / "comparison.csv", [rows[-1] for rows in histories])
    if common:
        last = max(common)
        plot_rows = [
            dict(index[last], method=method) for method, index in zip(methods, indexes)
        ]
        save(Path(output) / "matched_final.json", plot_rows)
        plot(Path(output) / "matched_final.json", Path(output) / "figures")
    return dict(common_round_byte_points=sorted(common), matched_rows=matched)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/clients")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--output", default="runs/native")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[
            "dpsgd",
            "dfedavg",
            "matcha",
            "relaysum",
            "dfedsam",
            "stair",
            "future_oracle",
        ],
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = compare(
        np.load(Path(args.data) / "dataset.npz"),
        load(Path(args.data) / "contacts.json"),
        load(args.config),
        args.output,
        args.methods,
    )
    print({"common_round_byte_points": result["common_round_byte_points"]})


if __name__ == "__main__":
    main()
