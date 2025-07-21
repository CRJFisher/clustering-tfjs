import numpy as np
import json
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian
import scipy.sparse as sp

# Load data
with open('./test/fixtures/spectral/blobs_n2_knn.json', 'r') as f:
    data = json.load(f)

X = np.array(data['X'])

# Create affinity matrix
affinity = kneighbors_graph(X, n_neighbors=10, mode='connectivity', include_self=True)
affinity = 0.5 * (affinity + affinity.T)

print("Affinity matrix properties:")
print(f"  Shape: {affinity.shape}")
print(f"  Density: {affinity.nnz / (affinity.shape[0] * affinity.shape[1]):.3f}")
print(f"  Min/max values: {affinity.min()}, {affinity.max()}")

# Different ways to compute Laplacian
print("\nDifferent Laplacian computations:")

# 1. Scipy normalized Laplacian (what sklearn uses)
L_norm, dd = laplacian(affinity, normed=True, return_diag=True)
print(f"\n1. Scipy normalized Laplacian:")
print(f"   Diagonal values: min={L_norm.diagonal().min():.6f}, max={L_norm.diagonal().max():.6f}")

# 2. Unnormalized Laplacian
L_unnorm = laplacian(affinity, normed=False)
print(f"\n2. Unnormalized Laplacian:")
print(f"   Diagonal values: min={L_unnorm.diagonal().min():.6f}, max={L_unnorm.diagonal().max():.6f}")

# 3. Random walk Laplacian
degrees = np.array(affinity.sum(axis=1)).flatten()
D = sp.diags(degrees)
L_rw = sp.eye(affinity.shape[0]) - sp.linalg.inv(D) @ affinity
print(f"\n3. Random walk Laplacian:")
print(f"   Diagonal values: min={L_rw.diagonal().min():.6f}, max={L_rw.diagonal().max():.6f}")

# Check if sklearn does any special preprocessing
from sklearn.manifold import spectral_embedding
from sklearn.utils.fixes import laplacian as sklearn_laplacian

# Get sklearn's Laplacian
L_sklearn, dd_sklearn = sklearn_laplacian(affinity, normed=True, return_diag=True)
print(f"\n4. sklearn's Laplacian function:")
print(f"   Same as scipy? {np.allclose(L_sklearn.toarray(), L_norm.toarray())}")

# The key might be in how sklearn handles the eigendecomposition
# Let's trace through what _spectral_embedding does
print("\n\nTracing sklearn's spectral_embedding:")

# Check the shift-invert parameters sklearn uses
from scipy.sparse.linalg import eigsh

# sklearn uses sigma=1.0 on -L
L_neg = -L_norm
try:
    # This is what sklearn does in shift-invert mode
    eigenvalues_si, eigenvectors_si = eigsh(L_neg, k=6, sigma=1.0, which='LM')
    
    print(f"\nShift-invert eigenvalues: {eigenvalues_si}")
    print(f"\nShift-invert eigenvectors unique values:")
    for i in range(3):
        idx = 5 - i  # sklearn reverses order
        unique = len(np.unique(np.round(eigenvectors_si[:, idx], 6)))
        print(f"  Eigenvector {i}: {unique} unique values")
        
    # Check if they're component indicators
    print("\nFirst few shift-invert eigenvector values:")
    for i in range(3):
        idx = 5 - i
        vec = eigenvectors_si[:, idx]
        print(f"  Vec {i} (first 10): {vec[:10]}")
        
except Exception as e:
    print(f"Shift-invert failed: {e}")

# Maybe the issue is elsewhere - let's check the full sklearn pipeline
print("\n\nDirect sklearn spectral_embedding call:")
embedding = spectral_embedding(
    affinity,
    n_components=2,
    drop_first=False,
    random_state=42
)

print(f"Embedding shape: {embedding.shape}")
for i in range(embedding.shape[1]):
    vec = embedding[:, i]
    unique = len(np.unique(np.round(vec, 6)))
    print(f"  Dimension {i}: {unique} unique values")
    if unique <= 5:
        print(f"    Values: {np.unique(np.round(vec, 6))}")