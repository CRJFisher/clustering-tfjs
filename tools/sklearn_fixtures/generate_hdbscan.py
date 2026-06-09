"""Generate reference fixtures for HDBSCAN via scikit-learn.

Sweeps ``min_cluster_size``, ``min_samples``, ``cluster_selection_epsilon`` and
``cluster_selection_method`` across blobs / moons / circles, plus a cosine case
supplied as a precomputed cosine distance matrix (scikit-learn HDBSCAN rejects
``metric='cosine'`` directly). Fixtures are written to ``__fixtures__/hdbscan``.

Beyond the dataset × parameter cross-product there are:

- **Parameter sweeps** — ``min_samples`` over three values on overlapping blobs
  and ``cluster_selection_epsilon`` over three values on a nested two-level
  blob hierarchy (the only regime where the epsilon merge changes the
  selection), each in both ``eom`` and ``leaf`` modes.
- **Degenerate cases** — an all-noise gaussian (every label ``-1``) and a
  single dense blob. With ``allow_single_cluster=False`` (the only mode the
  TypeScript estimator supports) HDBSCAN can never emit exactly one cluster:
  candidate clusters are created in sibling pairs, so a lone dense blob is
  root-only and comes back all-noise. Generation asserts these expected label
  sets so scikit-learn version drift fails loudly instead of committing wrong
  fixtures.

Every fixture records ``tie_free`` / ``min_mst_gap``, computed from the gaps
between sorted single-linkage merge distances (= mutual-reachability MST edge
weights). Distinct weights make the MST — and hence the whole flat clustering —
unique, so ``tie_free`` fixtures support exact label/probability assertions;
tie-bound fixtures only support tolerance-based parity, because numpy's
unstable argsort orders tied weights differently than the TypeScript Prim
implementation.

HDBSCAN is fully deterministic (no ``random_state``), so labels and
probabilities can be asserted with tight tolerances.

Versions are pinned in ``requirements.txt`` (scikit-learn 1.3.2, numpy 1.24.4);
tie structure and exact probabilities are version-sensitive.

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

# min_samples sweep: three values, both selection methods, on blobs that
# overlap enough for min_samples to matter. min_samples=2 keeps core
# distances below saturation, making that pair of fixtures tie-free.
MIN_SAMPLES_SWEEP: List[int] = [2, 5, 10]

# cluster_selection_epsilon sweep: three values, both methods, on the nested
# dataset where each epsilon visibly coarsens the selection (5 -> 3 -> 2
# clusters under eom).
EPSILON_SWEEP: List[float] = [0.8, 1.5, 3.0]

# Relative weight-gap below which two MST edge weights count as tied. Far
# above float64 noise between numpy and plain-JS distance computation, far
# below any genuine structural gap observed in the fixtures (>= 1e-7).
TIE_REL_MARGIN = 1e-9


def tie_info(model: Any) -> Dict[str, Any]:
    """Tie-freedom of the fitted model's MST edge weights (see module doc)."""
    slt = np.asarray(model._single_linkage_tree_)
    values = sorted(float(r["value"]) for r in slt)
    gaps = [b - a for a, b in zip(values, values[1:])]
    min_gap = min(gaps) if gaps else float("inf")
    scale = max(values[-1], 1e-300) if values else 1.0
    return {
        "tie_free": bool(min_gap > TIE_REL_MARGIN * scale),
        "min_mst_gap": float(min_gap),
    }


def fit_dump(
    name: str,
    combo: Dict[str, Any],
    *,
    X: np.ndarray | None = None,
    distance_matrix: np.ndarray | None = None,
    metric: str | None = None,
) -> Dict[str, Any]:
    if metric is None:
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
        **tie_info(model),
    }
    if distance_matrix is not None:
        fixture["distance_matrix"] = distance_matrix.astype(float).tolist()
    else:
        assert X is not None
        fixture["X"] = X.astype(float).tolist()
    return fixture


def combo_filename(ds_name: str, combo: Dict[str, Any]) -> str:
    ms = "def" if combo["min_samples"] is None else str(combo["min_samples"])
    return (
        f"{ds_name}_mcs{combo['min_cluster_size']}_ms{ms}"
        f"_{combo['method']}_eps{combo['eps']}.json"
    )


def write_parameter_sweeps() -> None:
    """min_samples and cluster_selection_epsilon sweeps, eom + leaf each."""
    X_overlap, _ = datasets.make_blobs(
        n_samples=100, centers=3, cluster_std=1.3, random_state=7
    )
    for ms in MIN_SAMPLES_SWEEP:
        for method in ("eom", "leaf"):
            combo = {
                "min_cluster_size": 10,
                "min_samples": ms,
                "method": method,
                "eps": 0.0,
            }
            fixture = fit_dump("blobs_overlap", combo, X=X_overlap)
            fname = combo_filename("blobs_overlap", combo)
            (OUT_DIR / fname).write_text(json.dumps(fixture, indent=2))

    # Nested two-level hierarchy: two pairs of close blobs. Leaf clusters are
    # born well below the swept epsilons, so the epsilon merge genuinely
    # coarsens the selection here.
    rng = np.random.RandomState(5)
    centers = [(0.0, 0.0), (2.2, 0.0), (10.0, 0.0), (12.2, 0.0)]
    X_nested = np.vstack([np.asarray(c) + 0.35 * rng.randn(25, 2) for c in centers])

    baseline: Dict[str, np.ndarray] = {}
    for method in ("eom", "leaf"):
        base = HDBSCAN(
            min_cluster_size=8,
            min_samples=2,
            cluster_selection_method=method,
            cluster_selection_epsilon=0.0,
        )
        baseline[method] = base.fit_predict(X_nested)

    for eps in EPSILON_SWEEP:
        for method in ("eom", "leaf"):
            combo = {
                "min_cluster_size": 8,
                "min_samples": 2,
                "method": method,
                "eps": eps,
            }
            fixture = fit_dump("nested", combo, X=X_nested)
            assert fixture["tie_free"], (
                f"nested eps={eps} {method}: expected a tie-free tree"
            )
            assert fixture["labels"] != baseline[method].astype(int).tolist(), (
                f"nested eps={eps} {method}: epsilon merge had no effect; "
                "the sweep would be vacuous"
            )
            fname = combo_filename("nested", combo)
            (OUT_DIR / fname).write_text(json.dumps(fixture, indent=2))


def write_degenerate_cases() -> None:
    """All-noise and single-dense-blob degenerate inputs (see module doc)."""
    X_noise = np.random.RandomState(31).normal(0.0, 1.0, size=(60, 2))
    for method in ("eom", "leaf"):
        combo = {
            "min_cluster_size": 30,
            "min_samples": None,
            "method": method,
            "eps": 0.0,
        }
        fixture = fit_dump("allnoise", combo, X=X_noise)
        assert all(l == -1 for l in fixture["labels"]), "allnoise: expected all -1"
        assert all(p == 0.0 for p in fixture["probabilities"])
        assert fixture["tie_free"], "allnoise: expected a tie-free tree"
        fname = combo_filename("allnoise", combo)
        (OUT_DIR / fname).write_text(json.dumps(fixture, indent=2))

    # A single dense blob. min_samples=1 makes mutual reachability equal the
    # raw (generically distinct) distances, so the fixture is tie-free.
    X_blob = np.random.RandomState(101).normal(0.0, 0.1, size=(50, 2))
    for method in ("eom", "leaf"):
        combo = {
            "min_cluster_size": 25,
            "min_samples": 1,
            "method": method,
            "eps": 0.0,
        }
        fixture = fit_dump("single_blob", combo, X=X_blob)
        assert all(l == -1 for l in fixture["labels"]), (
            "single_blob: with allow_single_cluster=False a lone dense blob "
            "is root-only and must come back all-noise"
        )
        assert fixture["tie_free"], "single_blob: expected a tie-free tree"
        fname = combo_filename("single_blob", combo)
        (OUT_DIR / fname).write_text(json.dumps(fixture, indent=2))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for X, ds_name in make_datasets():
        for combo in COMBOS:
            fixture = fit_dump(f"{ds_name}", combo, X=X)
            (OUT_DIR / combo_filename(ds_name, combo)).write_text(
                json.dumps(fixture, indent=2)
            )

    write_parameter_sweeps()
    write_degenerate_cases()

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

    # Manhattan (L1) native metric case on blobs.
    X_man, _ = datasets.make_blobs(
        n_samples=80, centers=3, cluster_std=0.55, random_state=42
    )
    man_combo = {
        "min_cluster_size": 5,
        "min_samples": None,
        "method": "eom",
        "eps": 0.0,
    }
    fixture = fit_dump("blobs_manhattan", man_combo, X=X_man, metric="manhattan")
    (OUT_DIR / "blobs_manhattan_mcs5.json").write_text(json.dumps(fixture, indent=2))

    n_files = len(list(OUT_DIR.glob("*.json")))
    print(f"Wrote {n_files} HDBSCAN fixtures to {OUT_DIR}")


if __name__ == "__main__":
    main()
