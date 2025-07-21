import numpy as np
import json
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.manifold import spectral_embedding
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh

fixture_name = 'circles_n2_rbf'

with open(f'./test/fixtures/spectral/{fixture_name}.json') as f:
    data = json.load(f)

X = np.array(data['X'])
affinity = rbf_kernel(X, gamma=1.0)

print("Checking embedding normalization:\n")

# Get raw eigenvectors
L = laplacian(affinity, normed=True)
eigenvalues, eigenvectors = eigh(L)

# sklearn's embedding
embedding = spectral_embedding(affinity, n_components=2, drop_first=False, random_state=42)

# Let's manually construct what we think sklearn does
n = len(X)

# 1. Constant vector
const_vec = np.ones(n) / np.sqrt(n)
const_scaled = const_vec * np.sqrt(1 - 0)  # eigenvalue = 0

# 2. Second eigenvector
eig1 = eigenvectors[:, 1]
eig1_scaled = eig1 * np.sqrt(1 - eigenvalues[1])

# Stack them
manual_embedding = np.column_stack([const_scaled, eig1_scaled])

print("1. Manual reconstruction:")
print(f"   Const dimension: first value = {manual_embedding[0, 0]:.10f}")
print(f"   Second dimension: range = [{manual_embedding[:, 1].min():.6f}, {manual_embedding[:, 1].max():.6f}]")

print("\n2. sklearn embedding:")
print(f"   Const dimension: first value = {embedding[0, 0]:.10f}")
print(f"   Second dimension: range = [{embedding[:, 1].min():.6f}, {embedding[:, 1].max():.6f}]")

# Check the ratio
ratio = embedding[0, 0] / manual_embedding[0, 0]
print(f"\n3. Scaling ratio: {ratio:.10f}")

# Maybe sklearn applies row normalization?
print("\n4. Checking different normalizations:")

# Try normalizing rows to unit length
manual_norm_rows = manual_embedding / np.linalg.norm(manual_embedding, axis=1, keepdims=True)
print(f"   Row normalization: const value = {manual_norm_rows[0, 0]:.10f}")

# Try normalizing so sum of squares = 1 for each column
manual_norm_cols = manual_embedding / np.sqrt(np.sum(manual_embedding**2, axis=0, keepdims=True))
print(f"   Column normalization: const value = {manual_norm_cols[0, 0]:.10f}")

# The key insight: sklearn might normalize the entire embedding matrix
total_norm = np.linalg.norm(manual_embedding, 'fro')  # Frobenius norm
manual_norm_fro = manual_embedding / total_norm
print(f"   Frobenius normalization: const value = {manual_norm_fro[0, 0]:.10f}")

# Or maybe it's related to sqrt(n)
manual_scaled_n = manual_embedding / np.sqrt(n)
print(f"   Scaled by sqrt(n): const value = {manual_scaled_n[0, 0]:.10f}")

# Actually, let's check the exact normalization sklearn uses
print(f"\n5. Reverse engineering sklearn's normalization:")
print(f"   sklearn const / our const = {ratio:.10f}")
print(f"   This ratio squared = {ratio**2:.10f}")
print(f"   sqrt(ratio) = {np.sqrt(ratio):.10f}")

# Check if it's related to the embedding dimension
print(f"\n6. Could it be related to embedding dimension?")
print(f"   1/sqrt(2) = {1/np.sqrt(2):.10f}")
print(f"   1/sqrt(n_components) = {1/np.sqrt(2):.10f}")
print(f"   Our ratio is approximately 1/5 = {1/5:.10f}")