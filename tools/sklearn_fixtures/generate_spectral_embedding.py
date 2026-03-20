"""Generate spectral embedding fixtures with intermediate values for numerical verification.

Usage:
    cd tools/sklearn_fixtures
    source .venv/bin/activate
    python generate_spectral_embedding.py --out-dir ../../test/fixtures/spectral
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn import datasets
from sklearn.cluster import SpectralClustering as SkSpectralClustering
from sklearn.manifold import spectral_embedding
from sklearn.metrics.pairwise import rbf_kernel


def generate_fixture(
    X: np.ndarray,
    gamma: float,
    n_clusters: int,
    random_state: int,
    out_path: Path,
) -> None:
    affinity = rbf_kernel(X, gamma=gamma)

    emb = spectral_embedding(
        affinity,
        n_components=n_clusters,
        drop_first=False,
        random_state=random_state,
    )

    model = SkSpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=random_state,
        assign_labels="kmeans",
    )
    labels = model.fit_predict(affinity)

    from scipy.sparse.csgraph import laplacian as sp_laplacian

    aff_no_diag = affinity.copy()
    np.fill_diagonal(aff_no_diag, 0)
    L_norm = sp_laplacian(aff_no_diag, normed=True)
    if hasattr(L_norm, "toarray"):
        L_norm = L_norm.toarray()
    eigenvalues = np.sort(np.linalg.eigvalsh(L_norm))[:n_clusters]

    fixture = {
        "X": X.astype(float).tolist(),
        "params": {
            "nClusters": n_clusters,
            "affinity": "rbf",
            "gamma": gamma,
            "randomState": random_state,
        },
        "labels": labels.astype(int).tolist(),
        "embedding": emb.astype(float).tolist(),
        "eigenvalues": eigenvalues.astype(float).tolist(),
    }

    out_path.write_text(json.dumps(fixture, indent=2))
    print(f"Fixture written to {out_path}")
    print(f"  Eigenvalues: {eigenvalues}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Fixture 1: Well-separated blobs (degenerate eigenvalues near 0)
    # Good for subspace comparison (Gram matrix test)
    X_blobs, _ = datasets.make_blobs(
        n_samples=20, centers=3, random_state=42, cluster_std=0.5
    )
    generate_fixture(X_blobs, gamma=1.0, n_clusters=3, random_state=42,
                     out_path=args.out_dir / "embedding_blobs_n3_rbf.json")

    # Fixture 2: Moons data with moderate gamma (non-degenerate eigenvalues)
    # Good for column-by-column comparison since eigenvalues are distinct
    X_moons, _ = datasets.make_moons(n_samples=30, noise=0.05, random_state=42)
    generate_fixture(X_moons, gamma=5.0, n_clusters=2, random_state=42,
                     out_path=args.out_dir / "embedding_moons_n2_rbf.json")


if __name__ == "__main__":
    main()
