"""Generate reference fixtures for HDBSCAN via scikit-learn.

Sweeps ``min_cluster_size``, ``min_samples``, ``cluster_selection_epsilon`` and
``cluster_selection_method`` across blobs / moons / circles, plus a cosine case
supplied as a precomputed cosine distance matrix (scikit-learn HDBSCAN rejects
``metric='cosine'`` directly). Fixtures are written to ``__fixtures__/hdbscan``.

HDBSCAN is fully deterministic (no ``random_state``), so labels and
probabilities can be asserted with tight tolerances.

Usage
-----
    cd tools/sklearn_fixtures
    .venv/bin/python generate_hdbscan.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn import datasets
from sklearn.cluster import HDBSCAN
from sklearn.metrics import pairwise_distances


OUT_DIR = Path(__file__).resolve().parents[2] / "__fixtures__" / "hdbscan"


def make_datasets(random_state: int = 42):
    X_blobs, _ = datasets.make_blobs(
        n_samples=80, centers=3, cluster_std=0.55, random_state=random_state
    )
    X_moons, _ = datasets.make_moons(
        n_samples=80, noise=0.07, random_state=random_state
    )
    X_circles, _ = datasets.make_circles(
        n_samples=80, factor=0.5, noise=0.06, random_state=random_state
    )
    return [(X_blobs, "blobs"), (X_moons, "moons"), (X_circles, "circles")]


COMBOS: List[Dict[str, Any]] = [
    {"min_cluster_size": 5, "min_samples": None, "method": "eom", "eps": 0.0},
    {"min_cluster_size": 10, "min_samples": None, "method": "eom", "eps": 0.0},
    {"min_cluster_size": 5, "min_samples": 3, "method": "eom", "eps": 0.0},
    {"min_cluster_size": 5, "min_samples": None, "method": "leaf", "eps": 0.0},
    {"min_cluster_size": 8, "min_samples": None, "method": "eom", "eps": 0.3},
]


def fit_dump(
    name: str,
    combo: Dict[str, Any],
    *,
    X: np.ndarray | None = None,
    distance_matrix: np.ndarray | None = None,
) -> Dict[str, Any]:
    metric = "precomputed" if distance_matrix is not None else "euclidean"
    model = HDBSCAN(
        min_cluster_size=combo["min_cluster_size"],
        min_samples=combo["min_samples"],
        cluster_selection_method=combo["method"],
        cluster_selection_epsilon=combo["eps"],
        metric=metric,
    )
    data = distance_matrix if distance_matrix is not None else X
    labels = model.fit_predict(data)

    # The exact single-linkage hierarchy lets the condensed-tree module be
    # validated for bit-exact parity, independent of MST tie-ordering.
    slt = np.asarray(model._single_linkage_tree_)
    hierarchy = [
        [int(r["left_node"]), int(r["right_node"]), float(r["value"]), int(r["cluster_size"])]
        for r in slt
    ]

    fixture: Dict[str, Any] = {
        "name": name,
        "params": {
            "min_cluster_size": combo["min_cluster_size"],
            "min_samples": combo["min_samples"],
            "cluster_selection_method": combo["method"],
            "cluster_selection_epsilon": combo["eps"],
            "metric": metric,
        },
        "labels": labels.astype(int).tolist(),
        "probabilities": model.probabilities_.astype(float).tolist(),
        "single_linkage_tree": hierarchy,
    }
    if distance_matrix is not None:
        fixture["distance_matrix"] = distance_matrix.astype(float).tolist()
    else:
        assert X is not None
        fixture["X"] = X.astype(float).tolist()
    return fixture


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for X, ds_name in make_datasets():
        for combo in COMBOS:
            ms = "def" if combo["min_samples"] is None else str(combo["min_samples"])
            fname = (
                f"{ds_name}_mcs{combo['min_cluster_size']}_ms{ms}"
                f"_{combo['method']}_eps{combo['eps']}.json"
            )
            fixture = fit_dump(f"{ds_name}", combo, X=X)
            (OUT_DIR / fname).write_text(json.dumps(fixture, indent=2))

    # Cosine case via a precomputed cosine distance matrix on angularly
    # separated directions (magnitude varies, direction carries the cluster) so
    # the cosine clustering is unambiguous.
    rng = np.random.RandomState(0)
    directions = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    blocks = []
    for d in directions:
        angular = d + 0.04 * rng.randn(30, 3)
        magnitudes = rng.uniform(0.5, 5.0, size=(30, 1))
        blocks.append(angular * magnitudes)
    X_blobs = np.vstack(blocks)
    cosine_D = pairwise_distances(X_blobs, metric="cosine")
    cosine_combo = {
        "min_cluster_size": 5,
        "min_samples": None,
        "method": "eom",
        "eps": 0.0,
    }
    fixture = fit_dump("blobs_cosine", cosine_combo, distance_matrix=cosine_D)
    (OUT_DIR / "blobs_cosine_precomputed_mcs5.json").write_text(
        json.dumps(fixture, indent=2)
    )

    n_files = len(list(OUT_DIR.glob("*.json")))
    print(f"Wrote {n_files} HDBSCAN fixtures to {OUT_DIR}")


if __name__ == "__main__":
    main()
