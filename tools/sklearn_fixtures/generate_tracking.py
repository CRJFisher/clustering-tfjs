"""Generate synthetic drifting-snapshot fixtures for cross-window cluster tracking.

scikit-learn has no native cluster tracker, so this is a synthetic deterministic
reference: drifting ``make_blobs`` snapshots are clustered per snapshot with
KMeans, cosine cost matrices are built between consecutive centroid sets, the
optimal assignment is taken from scipy ``linear_sum_assignment``, and the
pruned per-current assignment plus transition labels are derived with the same
rules the TypeScript ``track_clusters`` uses. Fixtures are written to
``__fixtures__/tracking``.

Usage
-----
    cd tools/sklearn_fixtures
    .venv/bin/python generate_tracking.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances


OUT_DIR = Path(__file__).resolve().parents[2] / "__fixtures__" / "tracking"

THRESHOLD = 0.5


def transitions_from(cost: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Mirror the TypeScript track_clusters classification exactly."""
    n_prev, n_curr = cost.shape
    max_cost = 1 - threshold

    row_ind, col_ind = linear_sum_assignment(cost)
    optimal = {int(r): int(c) for r, c in zip(row_ind, col_ind)}

    # Pruned per-current assignment.
    assignment = [-1] * n_curr
    for i, j in optimal.items():
        if cost[i, j] <= max_cost:
            assignment[j] = i

    out_curr = [sorted(int(j) for j in range(n_curr) if cost[i, j] <= max_cost) for i in range(n_prev)]
    in_prev = [sorted(int(i) for i in range(n_prev) if cost[i, j] <= max_cost) for j in range(n_curr)]

    transitions: List[Dict[str, Any]] = []
    for j in range(n_curr):
        if len(in_prev[j]) == 0:
            transitions.append({"type": "EMERGE", "prev": [], "curr": [j]})
        elif len(in_prev[j]) >= 2:
            transitions.append({"type": "MERGE", "prev": in_prev[j], "curr": [j]})
    for i in range(n_prev):
        if len(out_curr[i]) == 0:
            transitions.append({"type": "DIE", "prev": [i], "curr": []})
        elif len(out_curr[i]) >= 2:
            transitions.append({"type": "SPLIT", "prev": [i], "curr": out_curr[i]})
    for j in range(n_curr):
        if len(in_prev[j]) == 1:
            i = in_prev[j][0]
            if len(out_curr[i]) == 1:
                transitions.append({"type": "PERSIST", "prev": [i], "curr": [j]})

    return {
        "row_ind": [int(x) for x in row_ind],
        "col_ind": [int(x) for x in col_ind],
        "assignment": assignment,
        "transitions": transitions,
    }


def snapshot_centroids(n_centers: int, drift: float, seed: int) -> np.ndarray:
    centers = np.array(
        [[np.cos(2 * np.pi * k / n_centers), np.sin(2 * np.pi * k / n_centers)]
         for k in range(n_centers)]
    ) * 5.0 + drift
    X, _ = make_blobs(
        n_samples=20 * n_centers, centers=centers, cluster_std=0.4, random_state=seed
    )
    km = KMeans(n_clusters=n_centers, n_init=10, random_state=seed).fit(X)
    return km.cluster_centers_.astype(float)


def make_case(name: str, prev: np.ndarray, curr: np.ndarray) -> Dict[str, Any]:
    cost = pairwise_distances(prev, curr, metric="cosine")
    ref = transitions_from(cost, THRESHOLD)
    return {
        "name": name,
        "threshold": THRESHOLD,
        "prev_centroids": prev.tolist(),
        "curr_centroids": curr.tolist(),
        "cost_matrix": cost.astype(float).tolist(),
        **ref,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Square, mild drift -> mostly persist.
    prev3 = snapshot_centroids(3, 0.0, 1)
    curr3 = snapshot_centroids(3, 0.3, 2)
    (OUT_DIR / "drift_n3.json").write_text(
        json.dumps(make_case("drift_n3", prev3, curr3), indent=2)
    )

    # Rectangular: 3 prev -> 4 curr (an emergence).
    curr4 = snapshot_centroids(4, 0.2, 3)
    (OUT_DIR / "grow_3_to_4.json").write_text(
        json.dumps(make_case("grow_3_to_4", prev3, curr4), indent=2)
    )

    # Rectangular: 4 prev -> 2 curr (deaths / merges).
    prev4 = snapshot_centroids(4, 0.0, 4)
    curr2 = snapshot_centroids(2, 0.1, 5)
    (OUT_DIR / "shrink_4_to_2.json").write_text(
        json.dumps(make_case("shrink_4_to_2", prev4, curr2), indent=2)
    )

    print(f"Wrote tracking fixtures to {OUT_DIR}")


if __name__ == "__main__":
    main()
