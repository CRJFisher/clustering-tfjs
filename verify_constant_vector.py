import numpy as np
import json
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.manifold import spectral_embedding
from scipy.sparse.csgraph import laplacian, connected_components
from scipy.linalg import eigh

fixture_name = 'circles_n2_rbf'

with open(f'./test/fixtures/spectral/{fixture_name}.json') as f:
    data = json.load(f)

X = np.array(data['X'])
affinity = rbf_kernel(X, gamma=1.0)

print("Understanding sklearn's constant vector:\n")

# Check if graph is connected
n_components, _ = connected_components(affinity > 1e-10)
print(f"1. Graph connectivity: {n_components} component(s)")

# Get embeddings
emb_drop_true = spectral_embedding(affinity, n_components=2, drop_first=True, random_state=42)
emb_drop_false = spectral_embedding(affinity, n_components=2, drop_first=False, random_state=42)

print("\n2. Embedding analysis:")
print(f"   drop_first=True shape: {emb_drop_true.shape}")
print(f"   drop_first=False shape: {emb_drop_false.shape}")

# The key insight: when drop_first=False, sklearn adds a constant vector
print("\n3. First dimension analysis (drop_first=False):")
first_dim = emb_drop_false[:, 0]
print(f"   All values equal? {np.allclose(first_dim, first_dim[0])}")
print(f"   Value: {first_dim[0]:.10f}")
print(f"   This is: 1/sqrt(n) = 1/sqrt({len(X)}) = {1/np.sqrt(len(X)):.10f}")

# So sklearn's logic seems to be:
# 1. For connected graphs, the first eigenvector SHOULD be constant (all 1s normalized)
# 2. But due to numerical errors, it's not exactly constant
# 3. So sklearn explicitly creates a constant vector when drop_first=False

print("\n4. Verifying sklearn's approach:")
n = len(X)
# The constant eigenvector for a connected graph
const_eigvec = np.ones(n) / np.sqrt(n)
# Its eigenvalue is 0
const_eval = 0
# Apply diffusion map scaling: multiply by sqrt(1 - eigenvalue) = sqrt(1 - 0) = 1
scaled_const = const_eigvec * np.sqrt(1 - const_eval)

print(f"   Expected constant value: {scaled_const[0]:.10f}")
print(f"   sklearn's constant value: {first_dim[0]:.10f}")
print(f"   Match? {np.allclose(scaled_const[0], first_dim[0])}")

# The second dimension should be the first non-constant eigenvector
print("\n5. Second dimension analysis:")
L = laplacian(affinity, normed=True)
eigenvalues, eigenvectors = eigh(L)

# sklearn's second dimension when drop_first=False
second_dim = emb_drop_false[:, 1]
# Should correlate with the second eigenvector (index 1)
eig1_scaled = eigenvectors[:, 1] * np.sqrt(1 - eigenvalues[1])

corr = np.corrcoef(second_dim, eig1_scaled)[0, 1]
print(f"   Correlation with scaled eigenvector 1: {corr:.4f}")

# Summary
print("\n6. SOLUTION FOUND:")
print("   When drop_first=False (used by SpectralClustering):")
print("   - sklearn inserts a constant vector 1/sqrt(n) as first dimension")
print("   - Then adds eigenvectors starting from index 1")
print("   - Our implementation uses eigenvectors starting from index 0")
print("   - This mismatch causes different k-means results!")