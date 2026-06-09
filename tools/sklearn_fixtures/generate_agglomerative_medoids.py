"""Generate post-hoc medoid reference fixtures for AgglomerativeClustering.

For each dataset/params combination, fits scikit-learn AgglomerativeClustering,
then computes the medoid of every cluster — the sample closest to that
cluster's mean under the cluster's metric. Per-sample distances to the cluster
mean are stored alongside each expected medoid index so the TypeScript test can
disambiguate ties deterministically. Fixtures are written to
``__fixtures__/agglomerative`` with a ``medoids_`` prefix.

Usage
-----
    cd tools/sklearn_fixtures
    .venv/bin/python generate_agglomerative_medoids.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering


OUT_DIR = Path(__file__).resolve().parents[2] / "__fixtures__" / "agglomerative"


PARAM_GRID: List[Dict[str, Any]] = [
    {"n_clusters": 3, "linkage": "average", "metric": "euclidean"},
    {"n_clusters": 3, "linkage": "complete", "metric": "euclidean"},
    {"n_clusters": 3, "linkage": "average", "metric": "cosine"},
]


def make_datasets(random_state: int = 42):
    X_blobs, _ = datasets.make_blobs(
        n_samples=30, centers=3, random_state=random_state, cluster_std=0.60
    )
    X_moons, _ = datasets.make_moons(
        n_samples=30, noise=0.05, random_state=random_state
    )
    return [(X_blobs, "blobs"), (X_moons, "moons")]


def metric_distance(points: np.ndarray, mean: np.ndarray, metric: str) -> np.ndarray:
    if metric == "cosine":
        pn = points / (np.linalg.norm(points, axis=1, keepdims=True) + 1e-12)
        mn = mean / (np.linalg.norm(mean) + 1e-12)
        return 1.0 - pn @ mn
    if metric == "manhattan":
        return np.abs(points - mean).sum(axis=1)
    return np.linalg.norm(points - mean, axis=1)


def make_fixture(X: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
    model = AgglomerativeClustering(
        n_clusters=params["n_clusters"],
        linkage=params["linkage"],
        metric=params["metric"],
    )
    labels = model.fit_predict(X)
    n_clusters = params["n_clusters"]

    medoid_indices: List[int] = []
    # Per-cluster: distance of every sample in that cluster to the cluster mean.
    distances_to_mean: List[List[float]] = []
    member_indices: List[List[int]] = []

    for c in range(n_clusters):
        members = np.where(labels == c)[0]
        member_indices.append(members.astype(int).tolist())
        if members.size == 0:
            medoid_indices.append(-1)
            distances_to_mean.append([])
            continue
        cluster_points = X[members]
        mean = cluster_points.mean(axis=0)
        dists = metric_distance(cluster_points, mean, params["metric"])
        medoid_local = int(np.argmin(dists))
        medoid_indices.append(int(members[medoid_local]))
        distances_to_mean.append(dists.astype(float).tolist())

    return {
        "X": X.astype(float).tolist(),
        "params": params,
        "labels": labels.astype(int).tolist(),
        "medoid_indices": medoid_indices,
        "member_indices": member_indices,
        "distances_to_mean": distances_to_mean,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for X, ds_name in make_datasets():
        for p in PARAM_GRID:
            fixture = make_fixture(X, p)
            fname = f"medoids_{ds_name}_n{p['n_clusters']}_{p['linkage']}_{p['metric']}.json"
            (OUT_DIR / fname).write_text(json.dumps(fixture, indent=2))
    print(f"Wrote agglomerative medoid fixtures to {OUT_DIR}")


if __name__ == "__main__":
    main()
