#!/usr/bin/env python3
"""Extract spectral embedding from sklearn for comparison with our implementation."""

import json
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import spectral_embedding
import scipy.sparse

# Load the same fixture we're testing
with open('../../test/fixtures/spectral/circles_n2_knn.json', 'r') as f:
    fixture = json.load(f)

X = np.array(fixture['X'])
n_clusters = fixture['params']['nClusters']
affinity = fixture['params']['affinity']
n_neighbors = fixture['params']['nNeighbors']
random_state = fixture['params']['randomState']

print(f"Extracting sklearn spectral embedding for circles_n2_knn")
print(f"Dataset shape: {X.shape}")
print(f"Parameters: affinity={affinity}, n_neighbors={n_neighbors}, n_clusters={n_clusters}")
print(f"Expected ARI: 0.95+, Current ARI: 0.747\n")

# Build affinity matrix manually to match sklearn
if affinity == 'nearest_neighbors':
    connectivity = kneighbors_graph(
        X, n_neighbors=n_neighbors, include_self=True
    )
    affinity_matrix = 0.5 * (connectivity + connectivity.T)
elif affinity == 'rbf':
    gamma = 1.0 / X.shape[1] if 'gamma' not in fixture['params'] else fixture['params']['gamma']
    affinity_matrix = rbf_kernel(X, gamma=gamma)

print(f"Affinity matrix shape: {affinity_matrix.shape}")
if scipy.sparse.issparse(affinity_matrix):
    print(f"Affinity matrix min/max: {affinity_matrix.data.min():.6f} / {affinity_matrix.data.max():.6f}")
else:
    print(f"Affinity matrix min/max: {affinity_matrix.min():.6f} / {affinity_matrix.max():.6f}")

# Use sklearn's spectral_embedding function directly
# This returns the spectral embedding without the k-means step
embedding = spectral_embedding(
    affinity_matrix,
    n_components=n_clusters,
    eigen_solver='arpack' if scipy.sparse.issparse(affinity_matrix) else None,
    random_state=random_state,
    drop_first=True  # Drop the constant eigenvector
)

print(f"\nEmbedding shape: {embedding.shape}")
print(f"\nFirst 5 rows of embedding:")
for i in range(5):
    print(f"Row {i}: {', '.join(f'{v:.6f}' for v in embedding[i])}")

# Check embedding properties
print(f"\nEmbedding statistics:")
print(f"Min value: {embedding.min():.6f}")
print(f"Max value: {embedding.max():.6f}")
print(f"Mean value: {embedding.mean():.6f}")

# Check row norms
print(f"\nRow norms (first 5):")
for i in range(5):
    norm = np.linalg.norm(embedding[i])
    print(f"Row {i} norm: {norm:.6f}")

# Now run full spectral clustering to get labels
model = SpectralClustering(
    n_clusters=n_clusters,
    affinity=affinity,
    n_neighbors=n_neighbors,
    random_state=random_state,
    assign_labels='kmeans'
)
labels = model.fit_predict(X)

# Save for comparison
output = {
    'dataset': 'circles_n2_knn',
    'shape': list(embedding.shape),
    'embedding': embedding.tolist(),
    'labels': labels.tolist(),
    'affinity_min': float(affinity_matrix.min()),
    'affinity_max': float(affinity_matrix.max())
}

with open('sklearn_embedding.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nSaved sklearn embedding to sklearn_embedding.json")