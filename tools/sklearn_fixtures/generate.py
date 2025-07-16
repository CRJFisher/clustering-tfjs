"""Generate reference fixtures for AgglomerativeClustering via scikit-learn.

This script belongs to a tiny, isolated Python sub-project located under
`tools/sklearn_fixtures`.  The directory contains its own `requirements.txt`
so contributors can create a virtual environment without polluting the rest
of the repository.

Usage
-----
    # From the repo root
    cd tools/sklearn_fixtures
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

    python generate.py --out-dir ../../test/fixtures/agglomerative

The script is **one-shot only** – it's never run in CI. The resulting JSON
files feed the Jest tests in `test/clustering/agglomerative_reference.test.ts`.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering


DataSpec = Tuple[np.ndarray, str]


def generate_datasets(random_state: int = 42) -> List[DataSpec]:
    X_blobs, _ = datasets.make_blobs(
        n_samples=30, centers=3, random_state=random_state, cluster_std=0.60
    )
    X_moons, _ = datasets.make_moons(n_samples=30, noise=0.05, random_state=random_state)
    X_circles, _ = datasets.make_circles(
        n_samples=30, factor=0.5, noise=0.05, random_state=random_state
    )

    return [
        (X_blobs, "blobs"),
        (X_moons, "moons"),
        (X_circles, "circles"),
    ]


PARAM_GRID: List[Dict[str, Any]] = [
    {"n_clusters": 2, "linkage": "single", "metric": "euclidean"},
    {"n_clusters": 3, "linkage": "average", "metric": "euclidean"},
    {"n_clusters": 3, "linkage": "complete", "metric": "euclidean"},
    {"n_clusters": 3, "linkage": "ward", "metric": "euclidean"},
]


def dump_fixture(X: np.ndarray, params: Dict[str, Any], out_path: Path) -> None:
    model = AgglomerativeClustering(
        n_clusters=params["n_clusters"],
        linkage=params["linkage"],
        metric=params["metric"],
    )
    labels = model.fit_predict(X)

    fixture = {
        "X": X.astype(float).tolist(),
        "params": {
            "nClusters": params["n_clusters"],
            "linkage": params["linkage"],
            "metric": params["metric"],
        },
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
            fname = f"{ds_name}_n{p['n_clusters']}_{p['linkage']}_{p['metric']}.json"
            dump_fixture(X, p, args.out_dir / fname)

    print(
        f"Fixtures written to {args.out_dir}  "
        f"(total: {len(list(args.out_dir.glob('*.json')))} files)"
    )


if __name__ == "__main__":
    main()

