"""Time scikit-learn SpectralClustering with nearest-neighbor affinity.

Usage
-----
    cd tools/sklearn_fixtures
    source .venv/bin/activate
    python benchmark_spectral_knn.py --samples 10000 --features 10 --centers 5

The script prints a single JSON object so benchmark notes can record sklearn
timings beside the JavaScript dense and sparse spectral benchmark entries.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time

from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_blobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--features", type=int, default=10)
    parser.add_argument("--centers", type=int, default=5)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    X, _ = make_blobs(
        n_samples=args.samples,
        n_features=args.features,
        centers=args.centers,
        random_state=args.random_state,
    )

    model = SpectralClustering(
        n_clusters=args.centers,
        affinity="nearest_neighbors",
        n_neighbors=args.neighbors,
        random_state=args.random_state,
        assign_labels="kmeans",
    )

    start = time.perf_counter()
    labels = model.fit_predict(X)
    elapsed_ms = (time.perf_counter() - start) * 1000
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = max_rss / (1024 * 1024) if sys.platform == "darwin" else max_rss / 1024

    print(
        json.dumps(
            {
                "algorithm": "sklearn_spectral_nearest_neighbors",
                "samples": args.samples,
                "features": args.features,
                "centers": args.centers,
                "neighbors": args.neighbors,
                "elapsed_ms": round(elapsed_ms),
                "rss_mb": round(rss_mb),
                "labels": int(labels.shape[0]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
