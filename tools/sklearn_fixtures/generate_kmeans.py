"""Generate reference fixtures for KMeans centroids and nearest-centroid predict.

Emits, for several well-separated blob datasets, scikit-learn's
``cluster_centers_`` and the labels produced by ``KMeans.predict`` on a
held-out sample set. Fixtures are written to ``__fixtures__/kmeans``.

Well-separated blobs are used deliberately so both scikit-learn and the
TypeScript implementation converge to the same global optimum; cluster
assignments then agree up to a permutation of cluster ids, and centroids match
to high precision.

Usage
-----
    cd tools/sklearn_fixtures
    .venv/bin/python generate_kmeans.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


OUT_DIR = Path(__file__).resolve().parents[2] / "__fixtures__" / "kmeans"


CASES: List[Dict[str, Any]] = [
    {"name": "blobs_n3", "n_clusters": 3, "centers": 3, "std": 0.40},
    {"name": "blobs_n4", "n_clusters": 4, "centers": 4, "std": 0.50},
    {"name": "blobs_n5", "n_clusters": 5, "centers": 5, "std": 0.35},
]


def make_fixture(case: Dict[str, Any]) -> Dict[str, Any]:
    random_state = 42
    X, _ = make_blobs(
        n_samples=120,
        centers=case["centers"],
        cluster_std=case["std"],
        random_state=random_state,
    )
    x_test, _ = make_blobs(
        n_samples=30,
        centers=case["centers"],
        cluster_std=case["std"],
        random_state=random_state + 1,
    )

    model = KMeans(
        n_clusters=case["n_clusters"],
        n_init=10,
        random_state=random_state,
    )
    labels = model.fit_predict(X)
    predict_labels = model.predict(x_test)

    return {
        "name": case["name"],
        "params": {"n_clusters": case["n_clusters"], "random_state": random_state},
        "X": X.astype(float).tolist(),
        "labels": labels.astype(int).tolist(),
        "cluster_centers_": model.cluster_centers_.astype(float).tolist(),
        "inertia_": float(model.inertia_),
        "x_test": x_test.astype(float).tolist(),
        "predict_labels": predict_labels.astype(int).tolist(),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        fixture = make_fixture(case)
        (OUT_DIR / f"{case['name']}.json").write_text(json.dumps(fixture, indent=2))
    print(f"Wrote KMeans fixtures to {OUT_DIR}")


if __name__ == "__main__":
    main()
