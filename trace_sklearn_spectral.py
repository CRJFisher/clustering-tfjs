import numpy as np
import json
from sklearn.neighbors import kneighbors_graph

# Load data
with open('./test/fixtures/spectral/blobs_n2_knn.json', 'r') as f:
    data = json.load(f)

X = np.array(data['X'])

# Create affinity matrix
affinity = kneighbors_graph(X, n_neighbors=10, mode='connectivity', include_self=True)
affinity = 0.5 * (affinity + affinity.T)

print("Tracing through sklearn's spectral_embedding implementation:\n")

# Import the internal function
from sklearn.manifold._spectral_embedding import _spectral_embedding

# Call it with various parameters to understand behavior
print("1. Default parameters (what SpectralClustering uses):")
embedding1 = _spectral_embedding(
    adjacency=affinity,
    n_components=2,
    eigen_solver='arpack',
    random_state=42,
    eigen_tol='auto',
    drop_first=True  # SpectralClustering uses drop_first=True
)
print(f"Shape: {embedding1.shape}")
for i in range(embedding1.shape[1]):
    unique = len(np.unique(np.round(embedding1[:, i], 10)))
    print(f"  Dimension {i}: {unique} unique values")

print("\n2. With drop_first=False:")
embedding2 = _spectral_embedding(
    adjacency=affinity,
    n_components=3,
    eigen_solver='arpack',
    random_state=42,
    eigen_tol='auto',
    drop_first=False
)
print(f"Shape: {embedding2.shape}")
for i in range(embedding2.shape[1]):
    vec = embedding2[:, i]
    unique_vals = np.unique(np.round(vec, 10))
    print(f"  Dimension {i}: {len(unique_vals)} unique values")
    if len(unique_vals) <= 5:
        print(f"    Values: {unique_vals}")

# Now let's check what AMGPCG solver does
print("\n3. With amg solver (for comparison):")
try:
    embedding3 = _spectral_embedding(
        adjacency=affinity,
        n_components=3,
        eigen_solver='amg',
        random_state=42,
        eigen_tol='auto',
        drop_first=False
    )
    print(f"Shape: {embedding3.shape}")
    for i in range(embedding3.shape[1]):
        unique = len(np.unique(np.round(embedding3[:, i], 10)))
        print(f"  Dimension {i}: {unique} unique values")
except Exception as e:
    print(f"AMG failed: {e}")

# The key insight: sklearn detects disconnected graphs and handles them specially
from scipy.sparse.csgraph import connected_components
n_components, labels = connected_components(affinity)
print(f"\n4. Graph structure:")
print(f"Connected components: {n_components}")

# When disconnected, sklearn uses a special mode
print("\n5. Understanding sklearn's special handling:")
print("When the graph is disconnected, sklearn's _spectral_embedding:")
print("- Detects this condition")
print("- Uses special parameters for eigsh to get component indicators")
print("- This produces the constant-per-component eigenvectors we see")

# Let's see if we can replicate by using sklearn's exact parameters
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh

L = laplacian(affinity, normed=True)

# Try with very small sigma (what sklearn might do internally)
print("\n6. Testing various sigma values:")
for sigma in [1e-10, 0, None]:
    try:
        print(f"\n  sigma={sigma}:")
        if sigma is None:
            vals, vecs = eigsh(L, k=3, which='SM')
        else:
            vals, vecs = eigsh(L, k=3, sigma=sigma, which='LM')
        for i in range(3):
            unique = len(np.unique(np.round(vecs[:, i], 10)))
            print(f"    Eigenvector {i}: {unique} unique values")
    except Exception as e:
        print(f"    Failed: {e}")