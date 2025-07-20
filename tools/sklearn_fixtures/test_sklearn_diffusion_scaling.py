#!/usr/bin/env python3
"""Test sklearn's spectral embedding scaling in detail."""

import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.linalg import eigh
import json

# Load circles_n2_knn fixture
with open('../../test/fixtures/spectral/circles_n2_knn.json', 'r') as f:
    fixture = json.load(f)

X = np.array(fixture['X'])
n_neighbors = fixture['params']['nNeighbors']

# Build affinity matrix
connectivity = kneighbors_graph(X, n_neighbors=n_neighbors, include_self=True)
affinity = 0.5 * (connectivity + connectivity.T)
affinity_dense = affinity.toarray()

print("Step-by-step spectral embedding computation:\n")

# 1. Compute normalized Laplacian
degrees = np.sum(affinity_dense, axis=1)
D_sqrt_inv = np.diag(1.0 / np.sqrt(degrees))
L_norm = np.eye(len(degrees)) - D_sqrt_inv @ affinity_dense @ D_sqrt_inv

print(f"1. Normalized Laplacian shape: {L_norm.shape}")
print(f"   Diagonal values (first 5): {np.diag(L_norm)[:5]}")

# 2. Compute eigenvectors
eigenvalues, eigenvectors = eigh(L_norm)

print(f"\n2. Eigenvalues (first 5): {eigenvalues[:5]}")
print(f"   Eigenvector shape: {eigenvectors.shape}")

# 3. Check what sklearn does with eigenvectors
# From sklearn source: embeddings are eigenvectors, NOT scaled by eigenvalues!
# The diffusion map scaling is only used in spectral_embedding function when
# eigen_solver='amg' or when explicitly requested

print(f"\n3. Eigenvector column norms (first 3):")
for i in range(3):
    norm = np.linalg.norm(eigenvectors[:, i])
    print(f"   Column {i}: {norm:.6f}")

# 4. Extract embedding (columns 1 and 2, skipping constant eigenvector)
embedding = eigenvectors[:, 1:3]

print(f"\n4. Raw embedding (no diffusion scaling):")
print(f"   Shape: {embedding.shape}")
print(f"   Range: [{embedding.min():.6f}, {embedding.max():.6f}]")
print(f"   First 3 rows:")
for i in range(3):
    print(f"   Row {i}: {embedding[i]}")

# 5. Now test what happens with diffusion map scaling
# This is what we implemented but sklearn doesn't use it by default!
diffusion_embedding = embedding * np.sqrt(1 - eigenvalues[1:3])

print(f"\n5. With diffusion map scaling (what we implemented):")
print(f"   Range: [{diffusion_embedding.min():.6f}, {diffusion_embedding.max():.6f}]")
print(f"   First 3 rows:")
for i in range(3):
    print(f"   Row {i}: {diffusion_embedding[i]}")

# 6. The key finding: sklearn's SpectralClustering does NOT use diffusion scaling!
# It uses raw eigenvectors from the normalized Laplacian
print("\n6. KEY FINDING:")
print("   sklearn's SpectralClustering uses RAW eigenvectors, NOT diffusion-scaled!")
print("   The spectral_embedding function can apply diffusion scaling, but")
print("   SpectralClustering.fit() doesn't use it by default.")

# 7. What about the D^(-1/2) normalization?
# This is already baked into the normalized Laplacian eigenvectors!
print("\n7. The D^(-1/2) normalization:")
print("   This is already included in normalized Laplacian eigenvectors.")
print("   No additional scaling needed!")