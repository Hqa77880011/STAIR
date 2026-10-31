"""Pure optimization worker; isolates incompatible Windows BLAS/OpenMP runtimes."""

import json
from pathlib import Path
import sys
import networkx as nx
import numpy as np


def matching_decomposition(edges, clients):
    """Partition a simple graph into edge-disjoint conflict-free matchings."""
    graph = nx.Graph()
    graph.add_nodes_from(range(clients))
    graph.add_edges_from(edges)
    colors = nx.coloring.greedy_color(nx.line_graph(graph), strategy="largest_first")
    return [
        sorted(e for e in colors if colors[e] == color)
        for color in sorted(set(colors.values()))
    ]


def matcha_design(edges, clients, budget=0.5):
    """Solve MATCHA's activation-probability and mixing-weight SDPs."""
    if not 0 < budget <= 1:
        raise ValueError("MATCHA communication budget must lie in (0,1]")
    local = Path(__file__).resolve().parents[2] / ".deps"
    if local.is_dir() and str(local) not in sys.path:
        sys.path.insert(0, str(local))
    import cvxpy as cp

    groups = matching_decomposition(edges, clients)
    if not groups:
        return dict(matchings=[], probability=[], alpha=0.0, spectral_bound=1.0)
    matrices = []
    for group in groups:
        laplacian = np.zeros((clients, clients))
        for i, j in group:
            laplacian[i, i] += 1
            laplacian[j, j] += 1
            laplacian[i, j] -= 1
            laplacian[j, i] -= 1
        matrices.append(laplacian)
    p = cp.Variable(len(groups))
    mean = sum(p[i] * value for i, value in enumerate(matrices))
    # Exact algebraic-connectivity SDP, also defined for a two-node graph.
    connectivity = cp.Variable()
    projector = np.eye(clients) - np.ones((clients, clients)) / clients
    problem = cp.Problem(
        cp.Maximize(connectivity),
        [
            p >= 0,
            p <= 1,
            cp.sum(p) <= budget * len(groups),
            mean - connectivity * projector >> 0,
        ],
    )
    problem.solve(solver="CLARABEL")
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError("MATCHA probability SDP: " + problem.status)
    probability = np.clip(p.value, 0, 1)
    mean = sum(value * v for value, v in zip(matrices, probability))
    variance = sum(value * v * (1 - v) for value, v in zip(matrices, probability))
    alpha, beta, rho = cp.Variable(), cp.Variable(), cp.Variable()
    identity = np.eye(clients)
    consensus = np.ones((clients, clients)) / clients
    bound = (
        (1 - rho) * identity
        - 2 * alpha * mean
        - consensus
        + beta * (mean @ mean + 2 * variance)
    )
    problem = cp.Problem(
        cp.Minimize(rho), [bound << 0, alpha >= 0, beta >= cp.square(alpha), rho >= 0]
    )
    problem.solve(solver="CLARABEL")
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError("MATCHA weight SDP: " + problem.status)
    return dict(
        matchings=groups,
        probability=probability.tolist(),
        alpha=float(alpha.value),
        spectral_bound=float(rho.value),
    )


if __name__ == "__main__":
    arguments = json.loads(sys.stdin.read())
    print(json.dumps(matcha_design(**arguments)), flush=True)
