#!/usr/bin/env python3
"""Check how sklearn normalizes eigenvectors in spectral embedding."""

import numpy as np
from sklearn.manifold import spectral_embedding
from sklearn.neighbors import kneighbors_graph
import scipy.sparse
import scipy.linalg

# Create a simple test case
np.random.seed(42)
X = np.random.randn(10, 2)

# Build affinity matrix
connectivity = kneighbors_graph(X, n_neighbors=3, include_self=True)
affinity = 0.5 * (connectivity + connectivity.T)

print("Testing sklearn's eigenvector normalization\n")

# Get spectral embedding
embedding = spectral_embedding(
    affinity,
    n_components=2,
    drop_first=True
)

print(f"Embedding shape: {embedding.shape}")
print(f"Embedding range: [{embedding.min():.6f}, {embedding.max():.6f}]")

# Check row norms
print("\nRow norms:")
for i in range(min(5, len(embedding))):
    norm = np.linalg.norm(embedding[i])
    print(f"Row {i}: {norm:.6f}")

# Now let's manually compute the embedding to understand the normalization
# Convert to dense for easier manipulation
affinity_dense = affinity.toarray()

# Compute degree matrix
degrees = np.sum(affinity_dense, axis=1)
D = np.diag(degrees)

# Compute normalized Laplacian: L = D^(-1/2) * A * D^(-1/2)
D_sqrt_inv = np.diag(1.0 / np.sqrt(degrees))
L_norm = np.eye(len(degrees)) - D_sqrt_inv @ affinity_dense @ D_sqrt_inv

print("\n\nManual computation:")
print(f"Laplacian shape: {L_norm.shape}")
print(f"Laplacian diagonal: {np.diag(L_norm)[:5]}")

# Get eigenvectors
eigenvalues, eigenvectors = scipy.linalg.eigh(L_norm)

print(f"\nEigenvalues (first 5): {eigenvalues[:5]}")
print(f"Eigenvector shape: {eigenvectors.shape}")

# Check eigenvector normalization from scipy.linalg.eigh
print("\nEigenvector column norms (from scipy.linalg.eigh):")
for i in range(min(5, eigenvectors.shape[1])):
    col_norm = np.linalg.norm(eigenvectors[:, i])
    print(f"Column {i}: {col_norm:.6f}")

# sklearn uses columns 1:n_components+1 (skipping the constant eigenvector)
manual_embedding = eigenvectors[:, 1:3]

print(f"\nManual embedding shape: {manual_embedding.shape}")
print(f"Manual embedding range: [{manual_embedding.min():.6f}, {manual_embedding.max():.6f}]")

# Compare with sklearn
print("\nComparison:")
print(f"Sklearn max: {np.abs(embedding).max():.6f}")
print(f"Manual max: {np.abs(manual_embedding).max():.6f}")
print(f"Ratio: {np.abs(manual_embedding).max() / np.abs(embedding).max():.6f}")

# Check if sklearn applies additional scaling
# sklearn might normalize by sqrt(n) or apply other scaling
n = len(X)
print(f"\nPossible scaling factors:")
print(f"sqrt(n) = {np.sqrt(n):.6f}")
print(f"1/sqrt(n) = {1.0/np.sqrt(n):.6f}")

# Test if scaling by 1/sqrt(n) matches
scaled_manual = manual_embedding / np.sqrt(n)
print(f"\nManual embedding scaled by 1/sqrt(n):")
print(f"Range: [{scaled_manual.min():.6f}, {scaled_manual.max():.6f}]")
print(f"Max abs value: {np.abs(scaled_manual).max():.6f}")
print(f"Matches sklearn? {np.allclose(np.abs(scaled_manual).max(), np.abs(embedding).max(), rtol=0.1)}")