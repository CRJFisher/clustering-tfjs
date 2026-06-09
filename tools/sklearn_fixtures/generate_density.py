"""Generate reference fixtures for the density-clustering graph primitives.

Covers the minimum spanning tree (against scipy), per-point k-distance
(core distance), and mutual-reachability distances (computed directly from
the definitions in numpy). Fixtures are written to ``__fixtures__/density``.

Usage
-----
    cd tools/sklearn_fixtures
    .venv/bin/python generate_density.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist


OUT_DIR = Path(__file__).resolve().parents[2] / "__fixtures__" / "density"


def canonical_edges(mst) -> list:
    """Return MST edges as sorted [source, target, weight] with source < target."""
    coo = mst.tocoo()
    edges = []
    for r, c, w in zip(coo.row, coo.col, coo.data):
        s, t = (int(r), int(c)) if r < c else (int(c), int(r))
        edges.append([s, t, float(w)])
    edges.sort(key=lambda e: (e[0], e[1]))
    return edges


def make_mst_fixture(name: str, X: np.ndarray) -> dict:
    D = cdist(X, X)
    mst = minimum_spanning_tree(D)
    return {
        "name": name,
        "X": X.astype(float).tolist(),
        "edges": canonical_edges(mst),
        "total_weight": float(mst.sum()),
    }


def make_kdistance_fixture(name: str, X: np.ndarray, k_values: list) -> dict:
    D = cdist(X, X)
    # nearest-first distances per row, including self at column 0 (distance 0)
    nd = np.sort(D, axis=1)
    cores = {str(k): nd[:, k - 1].astype(float).tolist() for k in k_values}
    return {
        "name": name,
        "X": X.astype(float).tolist(),
        "neighbor_distances": nd.astype(float).tolist(),
        "k_values": k_values,
        "core_distances": cores,
    }


def make_mreach_fixture(name: str, X: np.ndarray, k: int) -> dict:
    D = cdist(X, X)
    nd = np.sort(D, axis=1)
    core = nd[:, k - 1]
    M = np.maximum(np.maximum(core[:, None], core[None, :]), D)
    return {
        "name": name,
        "X": X.astype(float).tolist(),
        "k": k,
        "core_distances": core.astype(float).tolist(),
        "distance_matrix": D.astype(float).tolist(),
        "mutual_reachability": M.astype(float).tolist(),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)

    # Small random point clouds with (almost surely) distinct pairwise
    # distances, so the minimum spanning tree is unique.
    datasets = {
        "small_2d": rng.rand(8, 2),
        "medium_2d": rng.rand(20, 2),
        "small_3d": rng.rand(12, 3),
    }

    for name, X in datasets.items():
        mst = make_mst_fixture(name, X)
        (OUT_DIR / f"mst_{name}.json").write_text(json.dumps(mst, indent=2))

        kd = make_kdistance_fixture(name, X, k_values=[1, 2, 3, 4])
        (OUT_DIR / f"kdistance_{name}.json").write_text(json.dumps(kd, indent=2))

        mr = make_mreach_fixture(name, X, k=3)
        (OUT_DIR / f"mreach_{name}.json").write_text(json.dumps(mr, indent=2))

    print(f"Wrote density fixtures to {OUT_DIR}")


if __name__ == "__main__":
    main()
