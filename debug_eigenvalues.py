import numpy as np
import json
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh

# Load the same test data
with open('test/fixtures/spectral/circles_n2_rbf.json', 'r') as f:
    data = json.load(f)

X = np.array(data['X'])
params = data['params']

# Compute affinity matrix
gamma = params.get('gamma', 1.0)
affinity = rbf_kernel(X, gamma=gamma)

# Compute Laplacian
lap, dd = laplacian(affinity, normed=True, return_diag=True)

# Get eigenvalues and eigenvectors
eigenvalues, eigenvectors = eigh(lap)

print("Eigenvalues (first 10):")
print(eigenvalues[:10])

print("\nSmallest eigenvalue:", eigenvalues[0])
print("Second smallest eigenvalue:", eigenvalues[1])

print("\nFirst eigenvector (first 5 values):")
print(eigenvectors[:5, 0])

print("\nFirst eigenvector statistics:")
print(f"  min: {eigenvectors[:, 0].min()}")
print(f"  max: {eigenvectors[:, 0].max()}")
print(f"  std: {eigenvectors[:, 0].std()}")

# Apply the D^{-1/2} normalization
embedding = eigenvectors.copy()
embedding = embedding / dd[:, np.newaxis]

print("\nAfter D^{-1/2} normalization:")
print("First column (first 5 values):", embedding[:5, 0])
print("First column statistics:")
print(f"  min: {embedding[:, 0].min()}")
print(f"  max: {embedding[:, 0].max()}")
print(f"  std: {embedding[:, 0].std()}")