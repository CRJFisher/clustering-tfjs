import numpy as np
import json
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.manifold import spectral_embedding
from scipy.sparse.csgraph import laplacian
from scipy.sparse import diags

fixture_name = 'circles_n2_rbf'

with open(f'./test/fixtures/spectral/{fixture_name}.json') as f:
    data = json.load(f)

X = np.array(data['X'])
affinity = rbf_kernel(X, gamma=1.0)

print("Understanding sklearn's exact scaling:\n")

# Get sklearn's embedding
embedding = spectral_embedding(affinity, n_components=2, drop_first=False, random_state=42)
sklearn_const = embedding[:, 0]

print(f"1. sklearn's first dimension (constant):")
print(f"   All values equal? {np.allclose(sklearn_const, sklearn_const[0])}")
print(f"   Value: {sklearn_const[0]:.10f}")
print(f"   L2 norm: {np.linalg.norm(sklearn_const):.10f}")

# Compute what the theoretical constant eigenvector should be
degrees = affinity.sum(axis=1)
D_sqrt_inv = diags(1.0 / np.sqrt(degrees))
const_unnorm = np.ones(len(X))
const_norm = D_sqrt_inv @ const_unnorm
const_norm = const_norm / np.linalg.norm(const_norm)

print(f"\n2. Theoretical constant eigenvector (D^(-1/2) * 1 normalized):")
print(f"   First 5 values: {const_norm[:5]}")
print(f"   L2 norm: {np.linalg.norm(const_norm):.10f}")

# Check the ratio
print(f"\n3. Scaling analysis:")
print(f"   sklearn value / theoretical value = {sklearn_const[0] / const_norm[0]:.10f}")

# Maybe sklearn applies additional normalization?
# Let's check if it's related to the number of samples
n = len(X)
print(f"\n4. Possible normalizations:")
print(f"   1/sqrt(n) = {1/np.sqrt(n):.10f}")
print(f"   sklearn / theoretical = {sklearn_const[0] / const_norm[0]:.10f}")

# Actually, let's trace through what spectral_embedding might do
# It seems like sklearn might be applying a different normalization
# Let's check the actual embedding values more carefully

print(f"\n5. Detailed sklearn embedding analysis:")
unique_vals = np.unique(np.round(sklearn_const, 10))
print(f"   Dimension 0 (constant): unique={len(unique_vals)}, std={sklearn_const.std():.10f}")
if len(unique_vals) < 5:
    print(f"   Unique values: {unique_vals}")
print(f"   Dimension 1: min={embedding[:, 1].min():.6f}, max={embedding[:, 1].max():.6f}")

# The key might be in how sklearn normalizes the embedding
# Let's check if there's a pattern
sum_squared = np.sum(embedding**2, axis=0)
print(f"\n6. Sum of squares for each dimension:")
print(f"   Dim 0: {sum_squared[0]:.6f}")
print(f"   Dim 1: {sum_squared[1]:.6f}")

# Maybe sklearn normalizes so that sum of squares = 1/n?
print(f"\n7. Checking normalization hypothesis:")
print(f"   Sum of squares * n:")
print(f"   Dim 0: {sum_squared[0] * n:.6f}")
print(f"   Dim 1: {sum_squared[1] * n:.6f}")