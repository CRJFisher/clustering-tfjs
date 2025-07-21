import numpy as np
import json
from sklearn.cluster._spectral import spectral_embedding
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh

# Load fixture
with open('./test/fixtures/spectral/circles_n2_rbf.json', 'r') as f:
    fixture = json.load(f)

X = np.array(fixture['X'])
gamma = fixture['params']['gamma']
n_clusters = fixture['params']['nClusters']

print("Investigating sklearn's spectral_embedding normalization")
print("=" * 60)

# Get affinity
affinity = rbf_kernel(X, gamma=gamma)

# Get sklearn's embedding
embedding_sklearn = spectral_embedding(
    affinity, 
    n_components=n_clusters,
    drop_first=False,
    eigen_solver='arpack'
)

# Manual computation
laplacian, dd = csgraph.laplacian(affinity, normed=True, return_diag=True)
eigenvalues, eigenvectors = eigsh(laplacian, k=n_clusters, which='SM')

# Try different normalizations
print("\n1. Raw eigenvectors (no scaling):")
print(f"   First row: {eigenvectors[0]}")
print(f"   L2 norm of first row: {np.linalg.norm(eigenvectors[0]):.6f}")

print("\n2. Diffusion map scaling (sqrt(1 - lambda)):")
scaling = np.sqrt(np.maximum(0, 1 - eigenvalues))
embedding_diffusion = eigenvectors * scaling
print(f"   Scaling factors: {scaling}")
print(f"   First row: {embedding_diffusion[0]}")
print(f"   L2 norm of first row: {np.linalg.norm(embedding_diffusion[0]):.6f}")

print("\n3. sklearn's embedding:")
print(f"   First row: {embedding_sklearn[0]}")
print(f"   L2 norm of first row: {np.linalg.norm(embedding_sklearn[0]):.6f}")

# Check if sklearn applies additional normalization
print("\n4. Checking for D^{-1/2} normalization:")
# In normalized Laplacian, the eigenvectors are already in the D^{-1/2} space
# To get back to original space, multiply by D^{-1/2}
d_sqrt_inv = 1.0 / np.sqrt(dd)
embedding_normalized = embedding_diffusion * d_sqrt_inv[:, np.newaxis]
print(f"   First row after D^{-1/2}: {embedding_normalized[0]}")
print(f"   Matches sklearn? {np.allclose(embedding_normalized, embedding_sklearn)}")

# Let's check the actual sklearn source
print("\n5. Checking sklearn's exact computation:")
# sklearn applies: embedding = eigenvectors * dd^{-1/2}
# where dd is the degree vector
embedding_sklearn_manual = eigenvectors * d_sqrt_inv[:, np.newaxis]
# Then applies diffusion scaling
embedding_sklearn_manual *= scaling

print(f"   Manual sklearn computation first row: {embedding_sklearn_manual[0]}")
print(f"   Matches sklearn? {np.allclose(embedding_sklearn_manual, embedding_sklearn)}")

# The key insight: sklearn's normalized laplacian eigenvectors are in D^{1/2} space
# To use them for clustering, they need to be transformed back with D^{-1/2}
print("\n6. Summary:")
print(f"   sklearn applies: eigenvectors * D^{{-1/2}} * sqrt(1 - eigenvalues)")
print(f"   We were applying: eigenvectors * sqrt(1 - eigenvalues)")
print(f"   Missing factor: D^{{-1/2}}")