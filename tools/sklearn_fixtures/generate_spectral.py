"""Generate reference fixtures for SpectralClustering via scikit-learn.

Usage (from repo root)::

    cd tools/sklearn_fixtures
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

    python generate_spectral.py --out-dir ../../test/fixtures/spectral

The produced JSON fixtures are consumed by Jest tests located at
`test/clustering/spectral_reference.test.ts`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn import datasets
from sklearn.cluster import SpectralClustering


DataSpec = Tuple[np.ndarray, str]


def generate_datasets(random_state: int = 42) -> List[DataSpec]:
    X_blobs, _ = datasets.make_blobs(
        n_samples=60, centers=3, random_state=random_state, cluster_std=0.5
    )
    X_moons, _ = datasets.make_moons(n_samples=60, noise=0.05, random_state=random_state)
    X_circles, _ = datasets.make_circles(
        n_samples=60, factor=0.5, noise=0.04, random_state=random_state
    )

    return [
        (X_blobs, "blobs"),
        (X_moons, "moons"),
        (X_circles, "circles"),
    ]


PARAM_GRID: List[Dict[str, Any]] = [
    # RBF affinity with default gamma (1.0) – mirror our default-ish setting
    {"n_clusters": 2, "affinity": "rbf", "gamma": 1.0},
    {"n_clusters": 3, "affinity": "rbf", "gamma": 1.0},
    # k-NN graph with default n_neighbors = 10
    {"n_clusters": 2, "affinity": "nearest_neighbors", "n_neighbors": 10},
    {"n_clusters": 3, "affinity": "nearest_neighbors", "n_neighbors": 10},
]


def dump_fixture(X: np.ndarray, params: Dict[str, Any], out_path: Path) -> None:
    # Build kwargs selectively to avoid passing None and violating sklearn's
    # param validation.
    kwargs: Dict[str, Any] = {
        "n_clusters": params["n_clusters"],
        "affinity": params["affinity"],
        "random_state": 42,
        "assign_labels": "kmeans",
    }
    if "gamma" in params:
        kwargs["gamma"] = params["gamma"]
    if "n_neighbors" in params:
        kwargs["n_neighbors"] = params["n_neighbors"]

    model = SpectralClustering(**kwargs)
    labels = model.fit_predict(X)

    param_dict: Dict[str, Any] = {
        "nClusters": params["n_clusters"],
        "affinity": params["affinity"],
        "randomState": 42,
    }
    if "gamma" in params:
        param_dict["gamma"] = params["gamma"]
    if "n_neighbors" in params:
        param_dict["nNeighbors"] = params["n_neighbors"]

    fixture = {
        "X": X.astype(float).tolist(),
        "params": param_dict,
        "labels": labels.astype(int).tolist(),
    }

    out_path.write_text(json.dumps(fixture, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    datasets = generate_datasets()

    for X, ds_name in datasets:
        for p in PARAM_GRID:
            fname_parts = [ds_name, f"n{p['n_clusters']}"]
            if p["affinity"] == "rbf":
                fname_parts.append("rbf")
            else:
                fname_parts.append("knn")
            fname = "_".join(fname_parts) + ".json"
            dump_fixture(X, p, args.out_dir / fname)

    print(
        f"Fixtures written to {args.out_dir} "
        f"(total: {len(list(args.out_dir.glob('*.json')))} files)"
    )


if __name__ == "__main__":
    main()
