#!/usr/bin/env python3
"""Extract spectral embedding from sklearn for comparison with our implementation."""

import json
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph
import scipy.linalg

# Load the same fixture we're testing
with open('../test/fixtures/spectral/circles_n2_knn.json', 'r') as f:
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

# Create a custom SpectralClustering to expose internals
class SpectralClusteringDebug(SpectralClustering):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_ = None
        
    def fit(self, X):
        # Run the parent fit
        super().fit(X)
        
        # Now extract the embedding by replicating sklearn's internal logic
        if self.affinity == 'nearest_neighbors':
            connectivity = kneighbors_graph(
                X, n_neighbors=self.n_neighbors, include_self=True, n_jobs=self.n_jobs
            )
            affinity_matrix = 0.5 * (connectivity + connectivity.T)
        elif self.affinity == 'rbf':
            affinity_matrix = rbf_kernel(X, gamma=self.gamma)
        else:
            affinity_matrix = self.affinity_matrix_
            
        # Compute normalized Laplacian
        from sklearn.manifold._spectral_embedding import _graph_laplacian_dense
        laplacian, dd = _graph_laplacian_dense(affinity_matrix, normed=True, return_diag=True)
        
        # Get eigenvectors
        print(f"Computing eigenvectors of Laplacian (shape: {laplacian.shape})")
        
        # sklearn uses scipy.linalg.eigh for the eigenvector computation
        eigenvalues, eigenvectors = scipy.linalg.eigh(laplacian)
        
        # Select smallest eigenvectors (excluding the constant one)
        # sklearn selects indices 1 to n_clusters (0-indexed)
        embedding = eigenvectors[:, 1:n_clusters+1]
        
        # sklearn does NOT normalize rows for kmeans assign_labels
        # (only for discretize, which we're not using)
        self.embedding_ = embedding
        
        return self

# Create model with same parameters
model = SpectralClusteringDebug(
    n_clusters=n_clusters,
    affinity=affinity,
    n_neighbors=n_neighbors,
    random_state=random_state,
    assign_labels='kmeans'  # Important: using kmeans, not discretize
)

# Fit and get embedding
model.fit(X)
embedding = model.embedding_
labels = model.labels_

print(f"Embedding shape: {embedding.shape}")
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

# Save for comparison
output = {
    'dataset': 'circles_n2_knn',
    'shape': list(embedding.shape),
    'embedding': embedding.tolist(),
    'labels': labels.tolist()
}

with open('sklearn_embedding.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nSaved sklearn embedding to sklearn_embedding.json")

# Also save intermediate values for debugging
print(f"\nDebugging info:")
print(f"Affinity matrix shape: {model.affinity_matrix_.shape}")
print(f"Affinity matrix min/max: {model.affinity_matrix_.min():.6f} / {model.affinity_matrix_.max():.6f}")