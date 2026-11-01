"""Small native-optimizer gradient and persistence checks."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from stair import native
from stair.common import load, save

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/native_validation"
OUT.mkdir(parents=True, exist_ok=True)
config = load(ROOT / "configs/default.json")
config.update(
    clients=2,
    classes=2,
    rounds=2,
    batch_size=4,
    local_steps=1,
    matcha_budget=1.0,
    success_probability=1.0,
    cost=0.0,
    initial_age=1.0,
    seed=17,
)
rng = np.random.default_rng(17)
data = dict(
    x=rng.integers(0, 256, (12, 8, 8, 3), dtype=np.uint8),
    y=np.tile([0, 1], 6),
    client=np.repeat([0, 1], 6),
    split=np.tile(["train"] * 4 + ["val"] * 2, 2),
    test_x=rng.integers(0, 256, (4, 8, 8, 3), dtype=np.uint8),
    test_y=np.tile([0, 1], 2),
    test_client=np.repeat([0, 1], 2),
)
np.savez_compressed(OUT / "dataset.npz", **data)
contacts = [[(0, 1)]] * 2
save(OUT / "contacts.json", contacts)
save(OUT / "config.json", config)

# MATCHA's SDPs: on a two-node edge at full budget, p=1 and alpha=1/2.
design = native.matcha_design([(0, 1)], 2, 1.0)
np.testing.assert_allclose(design["probability"], [1.0], atol=1e-5)
np.testing.assert_allclose(design["alpha"], 0.5, atol=2e-4)
partial = native.matcha_design([(0, 1), (1, 2)], 3, 0.5)
assert sum(partial["probability"]) <= 1.0001 and 0 <= partial["spectral_bound"] < 1
assert all(
    len({i for edge in group for i in edge}) == 2 * len(group)
    for group in partial["matchings"]
)
weights = native.metropolis([(0, 1), (1, 2)], 3)
torch.testing.assert_close(weights.sum(0), torch.ones(3))
torch.testing.assert_close(weights, weights.T)


# Execute the authors' actual RelaySum core with nontrivial message/count memory.
class Quadratic:
    def init_state(self):
        return torch.zeros(3, 1, 1)

    def grad(self, state):
        return state - torch.tensor([1.0, 2.0, 4.0]).reshape(3, 1, 1)


values = list(
    native.relay_core()(Quadratic(), native.TreeWorld(3, [(0, 1), (1, 2)]), 1.0, 2)
)
# Round 1 is [1, 7/3, 2]; round 2 recovers the global target 7/3 everywhere.
torch.testing.assert_close(values[-1].state, torch.full((3, 1, 1), 7 / 3))

# All five native baselines plus STAIR: real CNN backward, metrics and save/reload.
result = native.compare(data, contacts, config, OUT)
assert result["common_round_byte_points"]
final = load(OUT / "matched_final.json")
assert len(final) == 7 and len({(r["round"], r["model_bytes"]) for r in final}) == 1
assert (
    final[0]["model_bytes"]
    == 2 * final[0]["round"] * native.LearningTask(data, config).initial[0].numel() * 4
)
for row in load(OUT / "comparison.json"):
    assert np.isfinite(row["mean_accuracy"]) and row["gradient_calls"] == 2
    checkpoint = torch.load(OUT / row["method"] / "models.pt", weights_only=True)
    assert torch.isfinite(checkpoint["state"]).all()
    assert not torch.equal(
        checkpoint["state"], native.LearningTask(data, config).init_state()
    )
    restored = native.LearningTask(data, config)
    accuracy = restored.accuracy(checkpoint["state"])
    np.testing.assert_allclose(np.mean(accuracy), row["mean_accuracy"])

# DFedAvg really executes K local gradients; SAM really changes the second gradient.
config["local_steps"] = 2
_, task = native.run(data, contacts[:1], config, "dfedavg", OUT / "local_steps")
assert task.calls == 2 and all(
    norm > 0 and np.isfinite(norm) for norm in task.grad_norms
)
plain, sam = native.LearningTask(data, config), native.LearningTask(data, config)
assert not torch.allclose(
    plain.gradient(plain.init_state()), sam.gradient(sam.init_state(), sam=True)
)
oracle = native.future_contact_oracle([[(0, 1), (1, 2)], [(0, 1)]], 3)
assert oracle["schedule"][0] == ((1, 2),) and oracle["max_missed_run"] == 1
blocked = dict(config, model_byte_budget=0)
try:
    native.run(data, contacts[:1], blocked, "dfedavg", OUT / "budget")
except ValueError as error:
    assert "Model-byte budget" in str(error)
else:
    raise AssertionError("Native optimizer must enforce the actual model-byte budget")
failed_transport = dict(config, success_probability=0.0)
rows, _ = native.run(data, contacts[:1], failed_transport, "dfedavg", OUT / "transport")
assert rows[0]["model_bytes"] > 0 and rows[0]["harmful_exchange_rate"] == 0
try:
    native.run(data, [contacts[0], []], config, "relaysum", OUT / "invalid")
except ValueError as error:
    assert "static tree" in str(error)
else:
    raise AssertionError("Dynamic contacts must not be presented as canonical RelaySum")
print(
    "PASS native STAIR: D-PSGD/D-FedAvg/MATCHA two SDPs/upstream RelaySum/DFedSAM/STAIR/future-contact oracle, real gradients and reload, exact joint round/model-byte checkpoints, budget/transport accounting, eligibility and harmful mixing metrics"
)
