"""Generate reference fixtures for the PCA estimator via scikit-learn.

Uses ``sklearn.decomposition.PCA(svd_solver='full')``. Components are
sign-ambiguous per axis, so the TypeScript test compares under the ``svd_flip``
convention (largest-magnitude entry of each component made positive). Fixtures
are written to ``__fixtures__/pca``.

Usage
-----
    cd tools/sklearn_fixtures
    .venv/bin/python generate_pca.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA


OUT_DIR = Path(__file__).resolve().parents[2] / "__fixtures__" / "pca"


def make_case(name: str, X: np.ndarray, n_components: int) -> Dict[str, Any]:
    # Components and transform are stored raw (mutually consistent). The
    # TypeScript test aligns per-component sign before comparing, applying the
    # svd_flip sign-normalization convention by reference.
    model = PCA(n_components=n_components, svd_solver="full")
    transformed = model.fit_transform(X)
    return {
        "name": name,
        "n_components": n_components,
        "X": X.astype(float).tolist(),
        "components_": model.components_.astype(float).tolist(),
        "explained_variance_": model.explained_variance_.astype(float).tolist(),
        "mean_": model.mean_.astype(float).tolist(),
        "transform": transformed.astype(float).tolist(),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)

    iris = datasets.load_iris().data
    blobs, _ = datasets.make_blobs(
        n_samples=60, centers=4, n_features=5, random_state=7
    )
    correlated = rng.randn(50, 3) @ rng.randn(3, 6)  # rank-3 in 6-d

    cases: List[Dict[str, Any]] = [
        make_case("iris_2", iris, 2),
        make_case("iris_3", iris, 3),
        make_case("blobs5_3", blobs, 3),
        make_case("correlated_3", correlated, 3),
    ]

    for case in cases:
        (OUT_DIR / f"{case['name']}.json").write_text(json.dumps(case, indent=2))
    print(f"Wrote {len(cases)} PCA fixtures to {OUT_DIR}")


if __name__ == "__main__":
    main()
