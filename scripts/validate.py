"""CPU functional checks; generated images are not paper performance evidence."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from types import SimpleNamespace
import numpy as np
import torch
from stair import cli
from stair.core import allocation, matching
from stair.common import load, save, plot, table

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/validation"
OUT.mkdir(parents=True, exist_ok=True)
c = load(ROOT / "configs/default.json")
c.update(
    clients=4,
    rounds=2,
    local_epochs=1,
    batch_size=8,
    seed=17,
    success_probability=1.0,
    cost=0.0,
    novelty_min=0.01,
)
rng = np.random.default_rng(17)
# A persistent eligible chain serves two endpoints and charges only denied eligibility.
prov = np.eye(4, dtype=int)
selected, eligible, debt, _, _ = allocation(
    [(0, 1), (1, 2)],
    np.ones(4),
    np.ones(4),
    np.zeros(4),
    np.ones((4, 4)),
    prov,
    c,
    "stair",
    rng,
)
served = {i for e in selected for i in e}
assert len(selected) == 1 and eligible == {0, 1, 2} and debt[3] == 0
assert all(debt[i] == (0 if i in served else c["debt_step"]) for i in eligible)
assert matching([(0, 1), (1, 2), (2, 3)], {(0, 1): 3.0, (1, 2): 2.0, (2, 3): 1.0}) == [
    (0, 1),
    (2, 3),
]
# Actual forward, finite loss, gradients and an optimizer update.
torch.manual_seed(17)
model = cli.Network(3, 2)
model.train()
inp = torch.randn(4, 3, 8, 8)
old = {k: v.detach().clone() for k, v in model.named_parameters()}
loss = torch.nn.functional.cross_entropy(model(inp), torch.tensor([0, 1, 0, 1]))
assert torch.isfinite(loss)
loss.backward()
assert all(
    p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    for p in model.parameters()
)
torch.optim.SGD(model.parameters(), lr=0.01).step()
assert all(not torch.equal(old[k], p) for k, p in model.named_parameters())
model.eval()
before = model(inp).detach()
torch.save(model.state_dict(), OUT / "probe.pt")
clone = cli.Network(3, 2)
clone.load_state_dict(torch.load(OUT / "probe.pt", weights_only=True))
clone.eval()
torch.testing.assert_close(before, clone(inp), rtol=1e-6, atol=1e-7)
# External NPZ loader, train/val/test split, contacts and all method controls.
x = rng.integers(0, 256, size=(48, 8, 8, 3), dtype=np.uint8)
y = np.tile([0, 1], 24)
client = np.repeat(np.arange(4), 12)
test_x = rng.integers(0, 256, size=(16, 8, 8, 3), dtype=np.uint8)
np.savez(
    OUT / "input.npz",
    x=x,
    y=y,
    client=client,
    test_x=test_x,
    test_y=np.tile([0, 1], 8),
    test_client=np.repeat(np.arange(4), 4),
)
save(OUT / "contacts.json", [[[0, 1], [1, 2], [2, 3]], [[0, 2], [1, 3]]])
a = SimpleNamespace(
    input=str(OUT / "input.npz"),
    contacts=str(OUT / "contacts.json"),
    data=str(OUT / "data"),
    output=str(OUT),
)
cli.prepare(a, c)
rows = [cli.run(a, c, m) for m in ["random", "age", "no_debt", "no_novelty", "stair"]]
save(OUT / "comparison.json", rows)
table(OUT / "comparison.csv", rows)
acc = load(OUT / "stair/client_accuracy.json")
np.testing.assert_allclose(rows[-1]["mean_accuracy"], np.mean(acc))
np.testing.assert_allclose(rows[-1]["bottom_decile_accuracy"], min(acc))
assert all(np.isfinite([v for k, v in r.items() if k != "method"]).all() for r in rows)
weights = torch.load(OUT / "stair/models.pt", weights_only=True)
restored = cli.Network(3, 2)
restored.load_state_dict(weights[0])
restored.eval()
assert torch.isfinite(restored(inp)).all()
plot(OUT / "comparison.json", OUT / "figures")
# The published larger architecture is an actual trainable choice, not an alias.
resnet = cli.build_model(1, 3, "resnet18")
resnet.train()
gray = torch.randn(4, 1, 32, 32)
key_names = ["conv1.weight", "layer4.1.conv2.weight", "fc.weight"]
previous = {k: dict(resnet.named_parameters())[k].detach().clone() for k in key_names}
objective = torch.nn.functional.cross_entropy(resnet(gray), torch.tensor([0, 1, 2, 0]))
assert torch.isfinite(objective)
objective.backward()
for name in key_names:
    gradient = dict(resnet.named_parameters())[name].grad
    assert (
        gradient is not None
        and torch.isfinite(gradient).all()
        and gradient.abs().sum() > 0
    )
torch.optim.SGD(resnet.parameters(), lr=0.01).step()
assert all(
    not torch.equal(previous[k], dict(resnet.named_parameters())[k]) for k in key_names
)
resnet.eval()
expected = resnet(gray).detach()
torch.save(resnet.state_dict(), OUT / "resnet_probe.pt")
loaded = cli.build_model(1, 3, "resnet18")
loaded.load_state_dict(torch.load(OUT / "resnet_probe.pt", weights_only=True))
loaded.eval()
torch.testing.assert_close(expected, loaded(gray), rtol=1e-6, atol=1e-7)
# Read real LEAF-shaped user shards and bin hand-computable contact intervals.
import prepare_leaf
import prepare_contacts

leaf = {
    "users": ["u0", "u1"],
    "user_data": {
        u: {
            "x": [([0.25] * 784), ([0.75] * 784), ([0.1] * 784), ([0.9] * 784)],
            "y": [0, 1, 0, 1],
        }
        for u in ["u0", "u1"]
    },
}
save(OUT / "leaf_train.json", leaf)
save(OUT / "leaf_test.json", leaf)
prepared = prepare_leaf.convert(
    [OUT / "leaf_train.json"], [OUT / "leaf_test.json"], OUT / "leaf.npz"
)
assert prepared["clients"] == 2 and np.load(OUT / "leaf.npz")["x"].shape == (8, 28, 28)
(OUT / "raw_contacts.csv").write_text(
    "sender,receiver,start,end\nA,B,0,2\nB,C,1.5,2.5\n", encoding="utf-8"
)
prepare_contacts.convert(OUT / "raw_contacts.csv", OUT / "binned.json", 3, 3, 1.0)
assert load(OUT / "binned.json") == [[[0, 1]], [[0, 1], [1, 2]], [[1, 2]]]
print(
    "PASS STAIR: matching/debt hand case, NPZ preparation, finite loss/all gradients/updates, save-load eval equality, 5 train/eval methods, metric aggregation, ResNet18 gradients/reload, LEAF/contact conversion and plots"
)
