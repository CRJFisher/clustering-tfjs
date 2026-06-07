"""Generate a standalone cosine pairwise-distance reference fixture.

Validates the cosine path of ``pairwise_distance_matrix`` independently of any
estimator, using ``sklearn.metrics.pairwise_distances(metric='cosine')``.
Fixture is written to ``__fixtures__/pairwise``.

Usage
-----
    cd tools/sklearn_fixtures
    .venv/bin/python generate_pairwise.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import pairwise_distances


OUT_DIR = Path(__file__).resolve().parents[2] / "__fixtures__" / "pairwise"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(7)
    # Non-trivial vectors with varied magnitudes (cosine ignores magnitude).
    X = rng.randn(15, 5) * rng.randint(1, 5, size=(15, 1))
    D = pairwise_distances(X, metric="cosine")
    fixture = {
        "X": X.astype(float).tolist(),
        "cosine_distances": D.astype(float).tolist(),
    }
    (OUT_DIR / "cosine.json").write_text(json.dumps(fixture, indent=2))
    print(f"Wrote cosine pairwise fixture to {OUT_DIR}")


if __name__ == "__main__":
    main()
