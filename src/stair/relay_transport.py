"""Deferred-sweep-v1 transport for the pinned RelaySum model core.

Physical contact rounds and completed SGD updates use separate clocks.
Failed delivery attempts consume payload and are retried."""
import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from .common import load, save, table
from .native import TreeWorld, LearningTask, relay_core


def run(task, tree, contacts, learning_rate, *, budget, success_probability=1., seed=17,
        client_budgets=None):
    """Transport actual frozen source messages under capacity-one physical slots."""
    initial = task.init_state()
    clients = initial.shape[0]
    if not contacts or clients < 2 or budget < 0 or not 0 <= success_probability <= 1:
        raise ValueError("Require contacts, at least two clients, nonnegative bytes and probability in [0,1]")
    tree = sorted({tuple(sorted(edge)) for edge in tree})
    if any(i < 0 or j >= clients for i, j in tree):
        raise ValueError("Tree endpoint lies outside the client set")
    world = TreeWorld(clients, tree)
    generator = relay_core()(task, world, learning_rate, len(contacts))
    state = next(generator).state.detach().clone()
    rng = np.random.default_rng(seed)
    client_budgets = [budget]*clients if client_budgets is None else list(client_budgets)
    if len(client_budgets) != clients or any(value < 0 for value in client_budgets):
        raise ValueError("Each client needs one model-payload budget")
    sent = [0]*clients
    rows, pending, provisional, committed_buffers = [], set(), None, None
    published = computed = model_bytes = counter_bytes = 0
    directed_bytes = state[0].numel()*state.element_size()
    for physical_round, available in enumerate(contacts, 1):
        available = {tuple(sorted(edge)) for edge in available}
        if any(i < 0 or j >= clients for i, j in available):
            raise ValueError("Contact endpoint lies outside the client set")
        if provisional is None and model_bytes+2*directed_bytes <= budget:
            committed_buffers = copy.deepcopy(task.buffers) if hasattr(task, "buffers") else None
            provisional = next(generator).state.detach().clone()
            # The pinned original generator exposes its actual model/count packets.
            frame = generator.gi_frame.f_locals
            packets = {edge: value.detach().clone() for edge, value in frame["new_messages"].items()}
            counts = dict(frame["new_counts"])
            pending = set(tree)
            computed += 1
        chosen, successful, used, attempts = [], [], set(), []
        for edge in sorted(available & pending):
            i, j = edge
            if i in used or j in used or model_bytes+2*directed_bytes > budget:
                continue
            if sent[i]+directed_bytes > client_budgets[i] or sent[j]+directed_bytes > client_budgets[j]:
                continue
            # Both directed vectors are really source-produced packets, not state proxies.
            actual = sum(packets[a, b].numel()*packets[a, b].element_size() for a, b in ((i, j), (j, i)))
            if actual != 2*directed_bytes:
                raise RuntimeError("The source packet size differs from its model vector")
            model_bytes += actual
            counter_bytes += 16  # Two int64 source counts; network framing/ACKs are not modeled.
            sent[i] += directed_bytes
            sent[j] += directed_bytes
            used.update(edge)
            chosen.append(edge)
            ok = bool(rng.random() < success_probability)
            attempts.append(dict(edge=edge, delivered=ok, model_bytes=actual,
                                 counts=[int(counts[i, j]), int(counts[j, i])],
                                 source_packet_norms=[float(packets[i, j].norm()), float(packets[j, i].norm())]))
            if ok:
                pending.remove(edge)
                successful.append(edge)
        if provisional is not None and not pending:
            state = provisional
            provisional = None
            published += 1
        rows.append(dict(round=physical_round, model_bytes=model_bytes, counter_bytes=counter_bytes,
                         total_payload_bytes=model_bytes+counter_bytes, logical_updates=published,
                         computed_updates=computed, selected=chosen, successful=successful,
                         pending_edges=sorted(pending), attempts=attempts, state_norm=float(state.norm())))
    if provisional is not None and committed_buffers is not None:
        task.buffers = committed_buffers  # Checkpoint state and BN correspond to published updates.
    return dict(protocol="deferred-sweep-v1", optimizer="original_relaysum_model",
                state=state, rows=rows, published_updates=published, computed_updates=computed,
                model_bytes=model_bytes, counter_bytes=counter_bytes, client_model_bytes=sent,
                normalization="original world_size with original initial reference",
                failed_attempt_policy="charge both vectors, retain frozen epoch and retry",
                interrupted_epoch=provisional is not None)


def common_points(histories):
    """Return only observed common physical-round/model-byte points, never interpolation."""
    indexed = {name: {(r["round"], r["model_bytes"]): r for r in rows}
               for name, rows in histories.items()}
    common = set.intersection(*(set(rows) for rows in indexed.values())) if indexed else set()
    return [dict(round=t, model_bytes=b, methods={name: rows[t, b] for name, rows in indexed.items()})
            for t, b in sorted(common)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Native dataset.npz/contacts.json directory")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tree", required=True, help="Registered static tree JSON; no future-contact tree fitting")
    parser.add_argument("--success-probability", type=float, default=1.)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load(args.config)
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    task = LearningTask(np.load(Path(args.data)/"dataset.npz"), config)
    result = run(task, load(args.tree), load(Path(args.data)/"contacts.json")[:config["rounds"]],
                 config["lr"], budget=config["model_byte_budget"],
                 success_probability=args.success_probability, seed=config["seed"],
                 client_budgets=config.get("client_model_byte_budgets"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    state = result.pop("state")
    result["accuracy_per_client"] = task.accuracy(state)
    result["validation_loss_per_client"] = task.validation(state)
    torch.save(dict(state=state, buffers=task.buffers, config=config), output/"checkpoint.pt")
    save(output/"result.json", result)
    table(output/"clock_bytes.csv", [{k: v for k, v in row.items() if k not in ("attempts", "selected", "successful", "pending_edges")}
                                     for row in result["rows"]])


if __name__ == "__main__":
    main()
