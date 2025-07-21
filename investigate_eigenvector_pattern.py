import numpy as np
import json
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian
from sklearn.cluster import SpectralClustering

# Load blobs_n2_knn data
with open('./test/fixtures/spectral/blobs_n2_knn.json', 'r') as f:
    data = json.load(f)

X = np.array(data['X'])

# Create affinity matrix
affinity = kneighbors_graph(X, n_neighbors=10, mode='connectivity', include_self=True)
affinity = 0.5 * (affinity + affinity.T)

# Get Laplacian
L, dd = laplacian(affinity, normed=True, return_diag=True)
L_dense = L.toarray()

# Standard eigendecomposition
eigenvalues, eigenvectors = np.linalg.eigh(L_dense)

print("Standard eigendecomposition:")
print(f"First 5 eigenvalues: {eigenvalues[:5]}")
print(f"\nFirst eigenvector unique values: {len(np.unique(np.round(eigenvectors[:, 0], 6)))}")
print(f"Second eigenvector unique values: {len(np.unique(np.round(eigenvectors[:, 1], 6)))}")
print(f"Third eigenvector unique values: {len(np.unique(np.round(eigenvectors[:, 2], 6)))}")

# Check if eigenvectors have a pattern
print("\nChecking eigenvector structure:")
for i in range(3):
    vec = eigenvectors[:, i]
    # Group by similar values
    rounded = np.round(vec, 8)
    unique_vals, counts = np.unique(rounded, return_counts=True)
    print(f"\nEigenvector {i}:")
    print(f"  Unique values: {len(unique_vals)}")
    if len(unique_vals) <= 10:
        for val, count in zip(unique_vals, counts):
            print(f"    {val:.6f}: {count} occurrences")

# Check what sklearn does differently
print("\n\nChecking sklearn's approach:")
# Run spectral clustering
sc = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=42)

# Access the internal spectral embedding
from sklearn.manifold import spectral_embedding
embedding = spectral_embedding(affinity, n_components=2, drop_first=False, random_state=42)

print(f"\nsklearn embedding shape: {embedding.shape}")
print("Unique values per dimension:")
for i in range(embedding.shape[1]):
    unique = len(np.unique(np.round(embedding[:, i], 6)))
    print(f"  Dimension {i}: {unique} unique values")

# Check if there's a transformation between standard and sklearn eigenvectors
print("\n\nChecking relationship between eigenvectors:")
# The key might be in the normalization or scaling
for i in range(3):
    std_vec = eigenvectors[:, i]
    # Try recovering by dividing by sqrt(dd)
    recovered = std_vec / np.sqrt(dd)
    unique_recovered = len(np.unique(np.round(recovered, 6)))
    print(f"Eigenvector {i} after recovery: {unique_recovered} unique values")
    
# What about the degree matrix?
print(f"\nDegree values (dd): min={dd.min():.6f}, max={dd.max():.6f}")
print(f"Unique degree values: {len(np.unique(dd))}")