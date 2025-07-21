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

print("Investigating sklearn's spectral_embedding behavior:\n")

# Test different parameters
print("1. Testing drop_first parameter:")
for drop_first in [True, False]:
    embedding = spectral_embedding(
        affinity,
        n_components=3,
        drop_first=drop_first,
        random_state=42
    )
    print(f"\n   drop_first={drop_first}, shape={embedding.shape}")
    for i in range(min(3, embedding.shape[1])):
        vec = embedding[:, i]
        unique = len(np.unique(np.round(vec, 8)))
        constant = "CONSTANT" if unique == 1 else f"{unique} values"
        print(f"   Dim {i}: {constant}, range=[{vec.min():.6f}, {vec.max():.6f}]")

# Get the Laplacian eigenvectors
L = laplacian(affinity, normed=True)
eigenvalues, eigenvectors = eigh(L)

print("\n2. Raw eigenvectors from Laplacian:")
for i in range(3):
    vec = eigenvectors[:, i]
    unique = len(np.unique(np.round(vec, 8)))
    print(f"   Eigenvector {i}: {unique} unique values, eigenvalue={eigenvalues[i]:.6e}")

print("\n3. Understanding the constant vector:")
# The first eigenvector should be constant for connected graphs
vec0 = eigenvectors[:, 0]
print(f"   First eigenvector std: {vec0.std():.6e}")
print(f"   First eigenvector range: [{vec0.min():.6f}, {vec0.max():.6f}]")
print(f"   Is approximately constant? {vec0.std() < 1e-6}")

# Check what spectral_embedding does internally
print("\n4. Reconstructing sklearn's transformation:")

# For drop_first=False with n_components=2:
# sklearn returns 2 components, but which eigenvectors?
embedding_df_false = spectral_embedding(affinity, n_components=2, drop_first=False, random_state=42)

# Check correlation with eigenvectors
print("\n   Correlation with raw eigenvectors:")
for i in range(2):
    emb_vec = embedding_df_false[:, i]
    for j in range(3):
        eig_vec = eigenvectors[:, j]
        # Check correlation (might be scaled/flipped)
        corr = np.corrcoef(emb_vec, eig_vec)[0, 1]
        if abs(corr) > 0.99:
            print(f"   Embedding dim {i} correlates with eigenvector {j} (corr={corr:.4f})")

# The key insight: sklearn might be using a different eigenvector selection
print("\n5. Checking sklearn's eigenvector selection logic:")
print("   If graph is connected, first eigenvector is constant")
print("   sklearn might be returning this constant vector as first dimension")
print("   when drop_first=False, which our implementation doesn't do")

# Let's verify by creating the constant eigenvector manually
n = len(X)
const_vec = np.ones(n) / np.sqrt(n)  # Normalized constant vector
print(f"\n   Manual constant vector: unique values = {len(np.unique(const_vec))}")
print(f"   Matches sklearn dim 0? {np.allclose(np.abs(embedding_df_false[:, 0]), np.abs(const_vec))}")

# Check if sklearn's first dim is actually related to the first eigenvector
scaled_const = const_vec * np.sqrt(1 - eigenvalues[0])  # Apply diffusion scaling
print(f"   Scaled constant vector norm: {np.linalg.norm(scaled_const):.6f}")
print(f"   sklearn dim 0 norm: {np.linalg.norm(embedding_df_false[:, 0]):.6f}")
print(f"   Ratio: {np.linalg.norm(embedding_df_false[:, 0]) / np.linalg.norm(scaled_const):.6f}")