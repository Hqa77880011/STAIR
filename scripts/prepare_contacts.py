"""Bin normalized contact CSV intervals into undirected client contacts."""

import argparse
import csv
import json
import math
from pathlib import Path


def convert(
    input_path, output, clients, rounds, round_seconds, start_time=0.0, node_ids=None
):
    """A contact [start,end) is present in every round interval it overlaps."""
    if clients < 2 or rounds < 1 or round_seconds <= 0:
        raise ValueError("Require clients>=2, rounds>=1 and positive round duration")
    with Path(input_path).open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    required = {"sender", "receiver", "start", "end"}
    if not records or not required <= set(records[0]):
        raise ValueError("CSV requires sender,receiver,start,end header columns")
    nodes = sorted({r[key] for r in records for key in ("sender", "receiver")})
    chosen = [str(v) for v in node_ids] if node_ids is not None else nodes[:clients]
    if len(chosen) != clients or len(set(chosen)) != clients:
        raise ValueError("Supply one distinct trace node ID per model client")
    mapping = {node: i for i, node in enumerate(chosen)}
    contacts = [set() for _ in range(rounds)]
    for record in records:
        left, right = record["sender"], record["receiver"]
        start, end = float(record["start"]), float(record["end"])
        if not math.isfinite(start + end) or end <= start:
            raise ValueError("Every contact interval must be finite and increasing")
        if left not in mapping or right not in mapping or left == right:
            continue
        first = max(0, math.floor((start - start_time) / round_seconds))
        last = min(rounds - 1, math.ceil((end - start_time) / round_seconds) - 1)
        edge = tuple(sorted((mapping[left], mapping[right])))
        for index in range(first, last + 1):
            contacts[index].add(edge)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps([sorted(edges) for edges in contacts]), encoding="utf-8"
    )
    output.with_suffix(".nodes.json").write_text(json.dumps(chosen), encoding="utf-8")
    return dict(
        clients=clients,
        rounds=rounds,
        contact_edges=sum(len(edges) for edges in contacts),
        output=str(output),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--clients", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--round-seconds", type=float, required=True)
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument(
        "--nodes", help="JSON list mapping client index to raw trace node"
    )
    args = parser.parse_args()
    nodes = (
        json.loads(Path(args.nodes).read_text(encoding="utf-8")) if args.nodes else None
    )
    print(
        json.dumps(
            convert(
                args.input,
                args.output,
                args.clients,
                args.rounds,
                args.round_seconds,
                args.start_time,
                nodes,
            )
        )
    )


if __name__ == "__main__":
    main()
