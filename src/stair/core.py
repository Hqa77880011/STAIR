import numpy as np


def matching(edges, priority):
    remaining = {tuple(sorted(e)) for e in edges}
    chosen = []
    while remaining:
        vertices = {v for edge in remaining for v in edge}
        nominated = {
            i: max(
                (e for e in remaining if i in e),
                key=lambda e: (priority[e], -e[0], -e[1]),
            )
            for i in vertices
        }
        mutual = sorted(
            e for e in remaining if nominated[e[0]] == e and nominated[e[1]] == e
        )
        chosen.extend(mutual)
        used = {v for e in mutual for v in e}
        remaining = {e for e in remaining if not set(e) & used}
    return chosen


def allocation(edges, recovery, age, debt, usefulness, provenance, c, method, rng):
    eligible = []
    priority = {}
    bids = {}
    novelty = {}
    for i, j in edges:
        for receiver, sender in [(i, j), (j, i)]:
            novelty[receiver, sender] = np.mean(
                provenance[sender] > provenance[receiver]
            )
            if method == "no_novelty":
                novelty[receiver, sender] = 1.0
            gain = usefulness[receiver, sender] * novelty[receiver, sender]
            debt_term = 0 if method == "no_debt" else c["beta_d"] * debt[receiver]
            bids[receiver, sender] = (
                c["success_probability"]
                * (
                    c["beta_r"] * recovery[receiver] * gain
                    + c["beta_a"]
                    * min(age[receiver] / c["age_max"], 1)
                    * novelty[receiver, sender]
                )
                + debt_term
                - c["beta_c"] * c["cost"]
            )
        if method in ("random", "age"):
            eligible.append((i, j))
            priority[i, j] = (
                float(rng.random()) if method == "random" else float(age[i] + age[j])
            )
        elif (
            min(bids[i, j], bids[j, i]) > 0
            and min(novelty[i, j], novelty[j, i]) >= c["novelty_min"]
            and min(usefulness[i, j], usefulness[j, i]) >= c["usefulness_min"]
        ):
            eligible.append((i, j))
            priority[i, j] = bids[i, j] + bids[j, i]
    selected = matching(eligible, priority)
    served = {v for e in selected for v in e}
    available = {v for e in eligible for v in e}
    new_debt = debt.copy()
    for i in range(len(debt)):
        if i in served:
            new_debt[i] = 0
        elif i in available:
            new_debt[i] = min(c["debt_max"], debt[i] + c["debt_step"])
    return selected, available, new_debt, novelty, bids
