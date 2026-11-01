"""Original RelaySum math with real autograd and explicit intermittent transport."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import torch
from stair.common import save, load
from stair.native import TreeWorld, relay_core
from stair.native import LearningTask
from stair.relay_transport import run, common_points

torch.set_num_threads(1)


class RegressionTask:
    def __init__(self):
        self.calls = 0
        self.norms = []

    def init_state(self):
        return torch.zeros(3, 2, 1)

    def grad(self, state):
        parameters = state.detach().clone().requires_grad_(True)
        x = torch.tensor([[1., 2.], [2., 1.], [1., -1.]])
        y = torch.tensor([1., -1., 2.])
        predictions = (parameters.squeeze(-1)*x).sum(-1)
        ((predictions-y).square().sum()/2).backward()
        self.calls += 1
        self.norms.append(float(parameters.grad.norm()))
        return parameters.grad.detach()


tree = [[0, 1], [1, 2]]
contacts = [[[0, 1]], [[1, 2]], [[0, 1]], [[1, 2]]]
task = RegressionTask()
result = run(task, tree, contacts, .1, budget=64)
reference_task = RegressionTask()
iterator = relay_core()(reference_task, TreeWorld(3, tree), .1, 2)
next(iterator)
next(iterator)
expected = next(iterator).state
torch.testing.assert_close(result["state"], expected)
assert result["published_updates"] == result["computed_updates"] == task.calls == 2
assert task.norms[0] > 0 and result["state"].norm() > 0
assert [r["logical_updates"] for r in result["rows"]] == [0, 1, 1, 2]
assert result["model_bytes"] == 64 and result["counter_bytes"] == 64
for row in result["rows"]:
    endpoints = [client for edge in row["selected"] for client in edge]
    assert len(endpoints) == len(set(endpoints))
lost = run(RegressionTask(), tree, contacts, .1, budget=64, success_probability=0)
assert lost["model_bytes"] > 0 and lost["published_updates"] == 0
assert lost["state"].norm() == 0 and lost["interrupted_epoch"]
limited = run(RegressionTask(), tree, contacts, .1, budget=15)
assert limited["model_bytes"] == 0 and limited["computed_updates"] == 0
one_packet = run(RegressionTask(), tree, contacts, .1, budget=16)
assert one_packet["model_bytes"] == 16 and one_packet["published_updates"] == 0
assert not common_points(dict(full=result["rows"], lost=lost["rows"]))[-1]["methods"]["lost"]["logical_updates"]
assert common_points(dict(a=[dict(round=1, model_bytes=1)], b=[dict(round=1, model_bytes=2)])) == []
output = Path(__file__).resolve().parents[1]/"runs/relay-transport-v1"
root = Path(__file__).resolve().parents[1]
config = load(root/"configs/default.json")
config.update(clients=2, classes=2, rounds=2, batch_size=4, seed=17)
rng = np.random.default_rng(17)
data = dict(x=rng.integers(0, 256, (12, 8, 8, 3), dtype=np.uint8), y=np.tile([0, 1], 6),
            client=np.repeat([0, 1], 6), split=np.tile(["train"]*4+["val"]*2, 2),
            test_x=rng.integers(0, 256, (4, 8, 8, 3), dtype=np.uint8),
            test_y=np.tile([0, 1], 2), test_client=np.repeat([0, 1], 2))
torch.manual_seed(config["seed"])
cnn = LearningTask(data, config)
learned = run(cnn, [[0, 1]], [[[0, 1]], [[0, 1]]], config["lr"], budget=config["model_byte_budget"])
assert learned["published_updates"] == 2 and all(np.isfinite(cnn.grad_norms)) and min(cnn.grad_norms) > 0
assert not torch.equal(learned["state"], cnn.init_state())
checkpoint = dict(state=learned["state"], buffers=cnn.buffers)
output.mkdir(parents=True, exist_ok=True)
(output/"data").mkdir(exist_ok=True)
np.savez_compressed(output/"data/dataset.npz", **data)
save(output/"data/contacts.json", [[[0, 1]], [[0, 1]]])
save(output/"config.json", config)
save(output/"tree.json", [[0, 1]])
torch.save(checkpoint, output/"cnn.pt")
restored = LearningTask(data, config)
state = restored.restore(torch.load(output/"cnn.pt", weights_only=True))
np.testing.assert_allclose(restored.accuracy(state), cnn.accuracy(learned["state"]), atol=1e-7)
save(output/"validation.json", dict(protocol=result["protocol"], gradient_norms=task.norms,
                                     actual_model_bytes=result["model_bytes"], counter_bytes=result["counter_bytes"],
                                     physical_clock=[r["round"] for r in result["rows"]],
                                     logical_clock=[r["logical_updates"] for r in result["rows"]],
                                     matches_original_state=True, failed_attempts_charged=True,
                                     actual_cnn_updates=learned["published_updates"], cnn_grad_norms=cnn.grad_norms))
print("PASS original RelaySum/autograd state equivalence, capacity-one contacts, two clocks, "
      "failed retries, exact model/count bytes, interrupted budget and no interpolated comparison")
