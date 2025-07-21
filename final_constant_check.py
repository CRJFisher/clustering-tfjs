import numpy as np
import json
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.manifold import spectral_embedding
from sklearn.cluster import SpectralClustering
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh

fixture_name = 'circles_n2_rbf'

with open(f'./test/fixtures/spectral/{fixture_name}.json') as f:
    data = json.load(f)

X = np.array(data['X'])
affinity = rbf_kernel(X, gamma=1.0)

print("Final check on constant vector scaling:\n")

# Get sklearn's embedding
embedding = spectral_embedding(affinity, n_components=2, drop_first=False, random_state=42)
const_dim = embedding[:, 0]

print(f"1. sklearn's constant dimension:")
print(f"   Value: {const_dim[0]:.10f}")
print(f"   L2 norm: {np.linalg.norm(const_dim):.10f}")

# Check different possible scalings
n = len(X)
print(f"\n2. Possible constant vector constructions:")
print(f"   1/sqrt(n) = {1/np.sqrt(n):.10f}")
print(f"   1/n = {1/n:.10f}")
print(f"   sqrt(1/n) = {np.sqrt(1/n):.10f}")
print(f"   1/sqrt(n) * 1/sqrt(n) = {1/n:.10f}")

# The actual value seems to be 1/sqrt(n) scaled by something
ratio = const_dim[0] * np.sqrt(n)
print(f"\n3. Scaling factor: {ratio:.10f}")
print(f"   This is approximately: 1/sqrt(eigenvalue_of_D^-1/2 * 1 * D^-1/2)")

# Let's check the degree matrix impact
from scipy.sparse import diags
degrees = affinity.sum(axis=1)
D_sqrt_inv = diags(1.0 / np.sqrt(degrees))

# The normalized Laplacian eigenvector for eigenvalue 0 is D^(1/2) * 1
const_unnormalized = np.ones(n)
const_normalized_laplacian = D_sqrt_inv @ const_unnormalized
const_normalized_laplacian /= np.linalg.norm(const_normalized_laplacian)

print(f"\n4. Normalized Laplacian constant eigenvector:")
print(f"   First value: {const_normalized_laplacian[0]:.10f}")
print(f"   Is constant? {np.allclose(const_normalized_laplacian, const_normalized_laplacian[0])}")

# For spectral embedding, sklearn uses the eigenvectors of the normalized Laplacian
# but then applies an additional transformation
print(f"\n5. Understanding sklearn's transformation:")

# Get the actual first eigenvector
L = laplacian(affinity, normed=True)
eigenvalues, eigenvectors = eigh(L)
first_eigvec = eigenvectors[:, 0]

# Check if sklearn is using a corrected version
print(f"   Actual first eigenvector std: {first_eigvec.std():.6f}")
print(f"   Is it approximately D^(-1/2) * 1? {np.corrcoef(first_eigvec, const_normalized_laplacian)[0,1]:.4f}")

# The key insight
print(f"\n6. THE KEY INSIGHT:")
print(f"   sklearn replaces the numerically computed first eigenvector")
print(f"   with the theoretical constant eigenvector for connected graphs")
print(f"   This constant is D^(-1/2) * 1 normalized, then scaled by sqrt(1-λ)")

# Verify this
const_theoretical = const_normalized_laplacian * np.sqrt(1 - 0)  # eigenvalue = 0
print(f"\n   Theoretical constant first value: {const_theoretical[0]:.10f}")
print(f"   sklearn's constant first value: {const_dim[0]:.10f}")
print(f"   Ratio: {const_dim[0] / const_theoretical[0]:.10f}")

# Final check with different fixtures to confirm
print(f"\n7. This explains why sklearn gets perfect results:")
print(f"   - It uses the theoretical constant eigenvector")
print(f"   - Our implementation uses the numerically computed one")
print(f"   - The numerical one has small variations due to floating point")
print(f"   - These variations affect k-means clustering!")