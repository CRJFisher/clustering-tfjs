import numpy as np
import json
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import spectral_embedding
from sklearn.cluster import SpectralClustering

# Load data
with open('./test/fixtures/spectral/blobs_n2_knn.json', 'r') as f:
    data = json.load(f)

X = np.array(data['X'])

# Create affinity matrix exactly as in fixture
affinity = kneighbors_graph(X, n_neighbors=10, mode='connectivity', include_self=True)
affinity = 0.5 * (affinity + affinity.T)

print("Testing sklearn's exact implementation:\n")

# 1. Direct spectral_embedding call
print("1. spectral_embedding function:")
embedding = spectral_embedding(
    affinity, 
    n_components=3, 
    drop_first=True,  # sklearn drops first by default
    random_state=42
)

print(f"Shape: {embedding.shape}")
for i in range(embedding.shape[1]):
    vec = embedding[:, i]
    unique_vals = np.unique(np.round(vec, 10))
    print(f"Dimension {i}: {len(unique_vals)} unique values")
    if len(unique_vals) <= 5:
        print(f"  Values: {unique_vals}")

# 2. Through SpectralClustering
print("\n2. SpectralClustering class:")
sc = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=42)
labels = sc.fit_predict(affinity)

# Check internal embedding if accessible
if hasattr(sc, 'affinity_matrix_'):
    print("SpectralClustering has affinity_matrix_")
    
# Let's trace what happens with drop_first=False
print("\n3. With drop_first=False:")
embedding_no_drop = spectral_embedding(
    affinity, 
    n_components=3, 
    drop_first=False,
    random_state=42
)

print(f"Shape: {embedding_no_drop.shape}")
for i in range(embedding_no_drop.shape[1]):
    vec = embedding_no_drop[:, i]
    unique_vals = np.unique(np.round(vec, 10))
    print(f"Dimension {i}: {len(unique_vals)} unique values")
    if len(unique_vals) <= 5:
        print(f"  Values: {unique_vals}")

# Check if sklearn uses any special mode
print("\n4. Checking sklearn internals:")
# Look at the source to understand what mode it uses
from sklearn.manifold._spectral_embedding import _spectral_embedding

# Try to access internal implementation
eigen_solver = 'arpack'  # sklearn default
print(f"sklearn uses eigen_solver='{eigen_solver}' by default")

# The key might be in the eigsh call parameters
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import laplacian

L_norm = laplacian(affinity, normed=True)

# Try different eigsh modes
print("\n5. Testing eigsh modes:")

# Standard eigsh on L
try:
    vals1, vecs1 = eigsh(L_norm, k=3, which='SM')
    print(f"eigsh(L, which='SM'): {[len(np.unique(np.round(v, 10))) for v in vecs1.T]} unique values")
except Exception as e:
    print(f"Failed: {e}")

# eigsh on -L with sigma
try:
    vals2, vecs2 = eigsh(-L_norm, k=3, sigma=1.0, which='LM')
    print(f"eigsh(-L, sigma=1.0): {[len(np.unique(np.round(v, 10))) for v in vecs2.T]} unique values")
except Exception as e:
    print(f"Failed: {e}")