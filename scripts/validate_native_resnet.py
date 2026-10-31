"""Native per-client ResNet normalization state and full checkpoint restore."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from stair import native
from stair.common import load

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/native_validation"
data = np.load(OUT / "dataset.npz")
config = load(OUT / "config.json")
config.update(architecture="resnet18", rounds=1, local_steps=1, batch_size=4)
rows, task = native.run(data, [[(0, 1)]], config, "dfedsam", OUT / "resnet")
checkpoint = torch.load(OUT / "resnet/dfedsam/models.pt", weights_only=True)
assert len(checkpoint["buffers"]) == 2
for client in checkpoint["buffers"]:
    assert all(
        int(value) == 1
        for name, value in client.items()
        if name.endswith("num_batches_tracked")
    )
assert not torch.equal(
    checkpoint["buffers"][0]["bn1.running_mean"],
    checkpoint["buffers"][1]["bn1.running_mean"],
)
fresh = native.LearningTask(data, config)
restored = fresh.restore(checkpoint)
np.testing.assert_allclose(
    task.validation(checkpoint["state"]), fresh.validation(restored), rtol=1e-6
)
np.testing.assert_allclose(np.mean(fresh.accuracy(restored)), rows[-1]["mean_accuracy"])
assert task.grad_norms[0] > 0 and torch.isfinite(restored).all()
print(
    "PASS native ResNet-18: real two-client SAM gradients, independent BatchNorm buffers, one running-stat update per SAM step, full vector+buffer checkpoint inference parity"
)
