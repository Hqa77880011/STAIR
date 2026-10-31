"""Diagnostic counterexamples, not an intermittent RelaySum baseline or benchmark."""

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stair.core import allocation

torch.set_num_threads(1)
source = ROOT / "upstream/relaysgd/paper-figures"
sys.path.insert(0, str(source))
spec = importlib.util.spec_from_file_location(
    "relay_protocol_probe", source / "algorithms.py"
)
upstream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upstream)


class ChangingNeighbors:
    """Deliberately naive neighbor substitution, used only to expose the gap."""

    num_workers = 3
    workers = range(3)
    max_degree = 1
    round = -1
    contacts = [[(0, 1)], [(1, 2)]]

    def neighbors(self, worker):
        return [
            j if i == worker else i
            for i, j in self.contacts[self.round]
            if worker in (i, j)
        ]


class PulseTask:
    def __init__(self, world):
        self.world = world

    def init_state(self):
        return torch.zeros(3, 1, 1)

    def grad(self, state):
        self.world.round += 1
        # A single actual source-worker update, then exactly zero gradients.
        return (
            torch.tensor([-3.0, 0.0, 0.0]).reshape(3, 1, 1)
            if self.world.round == 0
            else torch.zeros_like(state)
        )


world = ChangingNeighbors()
iterator = upstream.relaysum_model(PulseTask(world), world, 1.0, 2)
next(iterator)
first = next(iterator).state.detach().clone().flatten()
second = next(iterator).state.detach().clone().flatten()
torch.testing.assert_close(first, torch.tensor([1.0, 1.0, 0.0]))
torch.testing.assert_close(second, torch.full((3,), 1 / 3))
assert world.neighbors(0) == [] and not torch.equal(first[0], second[0])

# One canonical three-node tree iteration needs both edges at its center.
tree = [(0, 1), (1, 2)]
center_exchanges = sum(1 in edge for edge in tree)
assert center_exchanges == 2  # Cannot be one capacity-one physical round.

# A shared upper bound does not force equal actual bytes at a shared round.
config = json.loads((ROOT / "configs/default.json").read_text(encoding="utf-8"))
selected, eligible, *_ = allocation(
    [(0, 1)],
    np.zeros(2),
    np.zeros(2),
    np.zeros(2),
    np.full((2, 2), config["usefulness_prior"]),
    np.eye(2, dtype=int),
    config,
    "stair",
    np.random.default_rng(17),
)
assert selected == [] and not eligible
parameter_bytes = 4  # One float32 parameter, two directions per bilateral exchange.
stair_bytes, relay_bytes = 0, 2 * parameter_bytes
assert stair_bytes < relay_bytes < config["model_byte_budget"]

print(
    json.dumps(
        dict(
            diagnostic_not_baseline=True,
            upstream_core="paper-figures/algorithms.py:relaysum_model",
            contacts=world.contacts,
            state_after_round_1=first.tolist(),
            state_after_round_2=second.tolist(),
            unmatched_client_0=dict(
                gradient=0,
                received_messages=0,
                before=float(first[0]),
                after=float(second[0]),
            ),
            canonical_tree_center_exchanges=center_exchanges,
            equal_round_equal_cap=dict(
                rounds=1,
                cap=config["model_byte_budget"],
                stair_actual_bytes=stair_bytes,
                two_node_relay_actual_bytes=relay_bytes,
            ),
        ),
        indent=2,
    )
)
print(
    "PASS protocol diagnostics: naive dynamic neighbors change an unmatched zero-gradient worker; tree iteration exceeds capacity one; equal caps/rounds do not enforce equal bytes"
)
