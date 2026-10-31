"""Convert standard LEAF/FEMNIST user_data JSON shards into grouped image NPZ."""

import argparse
import json
from pathlib import Path

import numpy as np


def read_users(paths):
    """Merge LEAF shards without splitting one writer across model clients."""
    users = {}
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for user, samples in payload["user_data"].items():
            x = np.asarray(samples["x"], dtype=np.float32)
            y = np.asarray(samples["y"], dtype=np.int64)
            if not len(y):
                continue
            if x.size != len(y) * 28 * 28 or not np.isfinite(x).all():
                raise ValueError("FEMNIST x must contain finite 28x28 images")
            x = x.reshape(-1, 28, 28)
            if x.min() < 0 or x.max() > 1:
                raise ValueError("LEAF FEMNIST input pixels must be in [0, 1]")
            users.setdefault(str(user), []).append(
                (np.rint(x * 255).astype(np.uint8), y)
            )
    return {
        user: (
            np.concatenate([v[0] for v in chunks]),
            np.concatenate([v[1] for v in chunks]),
        )
        for user, chunks in users.items()
    }


def convert(train_paths, test_paths, output, min_train=4, min_test=1):
    """Keep shared writers with enough train and held-out test observations."""
    train, test = read_users(train_paths), read_users(test_paths)
    users = sorted(
        user
        for user in train.keys() & test.keys()
        if len(train[user][1]) >= min_train and len(test[user][1]) >= min_test
    )
    if not users:
        raise ValueError("No shared writer has enough training and test samples")
    arrays = {}
    for source, prefix in [(train, ""), (test, "test_")]:
        arrays[prefix + "x"] = np.concatenate([source[u][0] for u in users])
        arrays[prefix + "y"] = np.concatenate([source[u][1] for u in users])
        arrays[prefix + "client"] = np.concatenate(
            [np.full(len(source[u][1]), i, dtype=np.int64) for i, u in enumerate(users)]
        )
    arrays["client_ids"] = np.asarray(users)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    return dict(
        clients=len(users),
        train_samples=len(arrays["y"]),
        test_samples=len(arrays["test_y"]),
        output=str(output),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--test", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(convert(args.train, args.test, args.output)))


if __name__ == "__main__":
    main()
