#!/usr/bin/env python3
"""
Analyze what's really happening in the tests vs our manual runs.
"""

import numpy as np
import json
import subprocess
import os
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

def load_fixture(fixture_name):
    """Load a fixture file."""
    path = f"test/fixtures/spectral/{fixture_name}"
    with open(path, 'r') as f:
        return json.load(f)

def analyze_blobs_clustering():
    """Analyze the blobs clustering in detail."""
    fixture = load_fixture('blobs_n2_knn.json')
    X = np.array(fixture['X'])
    y_true = np.array(fixture['labels'])
    
    print("Data Analysis:")
    print(f"Shape: {X.shape}")
    print(f"True labels unique: {np.unique(y_true)}")
    print(f"Label distribution: {np.bincount(y_true)}")
    
    # Look at the data visually
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    
    # Plot 1: True labels
    plt.subplot(131)
    colors = ['red' if l == 0 else 'blue' for l in y_true]
    plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6)
    plt.title('True Labels')
    
    # Plot 2: What our implementation predicts (based on manual test)
    our_preds = [0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0,
                 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0,
                 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0]
    colors_ours = ['red' if l == 0 else 'blue' for l in our_preds]
    plt.subplot(132)
    plt.scatter(X[:, 0], X[:, 1], c=colors_ours, alpha=0.6)
    plt.title('Our Predictions (ARI=0.088)')
    
    # Plot 3: Run sklearn
    model = SpectralClustering(
        n_clusters=2,
        affinity='nearest_neighbors',
        n_neighbors=10,
        random_state=42,
        n_init=10
    )
    sklearn_preds = model.fit_predict(X)
    colors_sklearn = ['red' if l == 0 else 'blue' for l in sklearn_preds]
    plt.subplot(133)
    plt.scatter(X[:, 0], X[:, 1], c=colors_sklearn, alpha=0.6)
    ari_sklearn = adjusted_rand_score(y_true, sklearn_preds)
    plt.title(f'Sklearn Predictions (ARI={ari_sklearn:.3f})')
    
    plt.tight_layout()
    plt.savefig('blobs_clustering_comparison.png', dpi=150)
    print("\nSaved visualization to blobs_clustering_comparison.png")
    
    # Analyze the pattern
    print("\n\nPattern Analysis:")
    print("Our predictions pattern (first 20):", our_preds[:20])
    print("True labels pattern (first 20):", y_true[:20].tolist())
    print("Sklearn predictions (first 20):", sklearn_preds[:20].tolist())
    
    # Check if there's a simple mapping issue
    print("\n\nChecking label mapping:")
    # Try flipping our labels
    our_flipped = [1 - p for p in our_preds]
    ari_flipped = adjusted_rand_score(y_true, our_flipped)
    print(f"ARI with flipped labels: {ari_flipped:.3f}")
    
    # Check connectivity
    from sklearn.neighbors import kneighbors_graph
    connectivity = kneighbors_graph(X, n_neighbors=10, mode='connectivity', include_self=True)
    connectivity = 0.5 * (connectivity + connectivity.T)
    
    print(f"\nConnectivity matrix: shape={connectivity.shape}, nnz={connectivity.nnz}")
    
    # Check if there are disconnected components
    from scipy.sparse.csgraph import connected_components
    n_components, labels = connected_components(connectivity, directed=False)
    print(f"Number of connected components: {n_components}")
    if n_components > 1:
        print(f"Component sizes: {np.bincount(labels)}")
        print(f"Component labels: {labels}")

if __name__ == "__main__":
    analyze_blobs_clustering()