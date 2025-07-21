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

# Get sklearn embedding
embedding = spectral_embedding(affinity, n_components=2, drop_first=False, random_state=42)
sklearn_dim1 = embedding[:, 1]

# Get our eigenvector
L = laplacian(affinity, normed=True)
eigenvalues, eigenvectors = eigh(L)
our_eigvec1 = eigenvectors[:, 1]

print("Detailed comparison of second dimension:\n")

# They should be proportional
ratio = sklearn_dim1[0] / our_eigvec1[0]
print(f"1. Scaling ratio from first element: {ratio:.10f}")

# Check if this ratio is consistent
our_scaled = our_eigvec1 * ratio

print("\n2. Element-wise comparison (first 10):")
print(f"{'Index':<6} {'sklearn':<15} {'ours*ratio':<15} {'diff':<15}")
print("-" * 55)
for i in range(10):
    diff = sklearn_dim1[i] - our_scaled[i]
    print(f"{i:<6} {sklearn_dim1[i]:<15.10f} {our_scaled[i]:<15.10f} {diff:<15.10f}")

# Check if the difference is related to eigenvalue scaling
print(f"\n3. Eigenvalue analysis:")
print(f"   Eigenvalue 1: {eigenvalues[1]:.10f}")
print(f"   sqrt(1 - eigenvalue): {np.sqrt(1 - eigenvalues[1]):.10f}")

# Our full scaling
our_with_diffusion = our_eigvec1 * np.sqrt(1 - eigenvalues[1])
ratio2 = sklearn_dim1[0] / our_with_diffusion[0]
print(f"\n4. With diffusion scaling:")
print(f"   New ratio: {ratio2:.10f}")
print(f"   This ratio squared: {ratio2**2:.10f}")
print(f"   This is close to: {0.04187625:.10f} (which we saw before)")

# The key insight: sklearn might be applying additional normalization
# Let's check the norm of sklearn's vectors
print(f"\n5. Vector norms:")
print(f"   sklearn dim 1 norm: {np.linalg.norm(sklearn_dim1):.10f}")
print(f"   Our eigenvector norm: {np.linalg.norm(our_eigvec1):.10f}")
print(f"   Our with diffusion norm: {np.linalg.norm(our_with_diffusion):.10f}")

# What if sklearn normalizes differently?
print(f"\n6. Checking sklearn's normalization:")
print(f"   Sum of squares: {np.sum(sklearn_dim1**2):.10f}")
print(f"   This * n = {np.sum(sklearn_dim1**2) * len(X):.10f}")
print(f"   sqrt(sum of squares): {np.sqrt(np.sum(sklearn_dim1**2)):.10f}")