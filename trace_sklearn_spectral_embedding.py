import numpy as np
import json
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
import inspect

# Let's look at sklearn's spectral_embedding source
from sklearn.manifold import spectral_embedding as manifold_spectral_embedding
from sklearn.cluster._spectral import spectral_embedding as cluster_spectral_embedding

print("Checking if spectral embeddings are different:")
print("=" * 60)

# Load fixture
with open('./test/fixtures/spectral/circles_n2_rbf.json', 'r') as f:
    fixture = json.load(f)

X = np.array(fixture['X'])
gamma = fixture['params']['gamma']
n_clusters = fixture['params']['nClusters']

affinity = rbf_kernel(X, gamma=gamma)

# Compare the two spectral_embedding functions
embedding1 = manifold_spectral_embedding(affinity, n_components=n_clusters, drop_first=False)
embedding2 = cluster_spectral_embedding(affinity, n_components=n_clusters, drop_first=False)

print(f"\n1. sklearn.manifold.spectral_embedding:")
print(f"   First row: {embedding1[0]}")
print(f"   Shape: {embedding1.shape}")

print(f"\n2. sklearn.cluster._spectral.spectral_embedding:")  
print(f"   First row: {embedding2[0]}")
print(f"   Shape: {embedding2.shape}")

print(f"\n3. Are they the same? {np.allclose(embedding1, embedding2)}")

# Let's manually trace through what sklearn does
print("\n4. Manual trace of sklearn's computation:")

# Step 1: Compute normalized Laplacian
laplacian, dd = csgraph.laplacian(affinity, normed=True, return_diag=True)
print(f"   Degree diagonal (first 5): {dd[:5]}")

# Step 2: Eigendecomposition  
eigenvalues, eigenvectors = eigsh(laplacian, k=n_clusters, which='SM')
print(f"   Eigenvalues: {eigenvalues}")

# Step 3: What sklearn actually does (from reading the source)
# For norm_laplacian=True, sklearn does NOT apply D^{-1/2}
# It uses the eigenvectors directly from the normalized Laplacian
embedding_manual = eigenvectors.copy()

# Step 4: Apply diffusion maps scaling
lambdas = eigenvalues
embedding_manual = embedding_manual * np.sqrt(1 - lambdas)

print(f"\n   Manual embedding first row: {embedding_manual[0]}")
print(f"   Matches sklearn? {np.allclose(embedding_manual, embedding1, atol=1e-6)}")

# Actually, let's check if drop_first matters
print("\n5. Checking drop_first parameter:")
embedding_drop = cluster_spectral_embedding(affinity, n_components=n_clusters+1, drop_first=True)
print(f"   With drop_first=True, shape: {embedding_drop.shape}")
print(f"   First row: {embedding_drop[0]}")

# The issue might be in how we're interpreting the results
print("\n6. Direct SpectralClustering check:")
from sklearn.cluster import SpectralClustering
sc = SpectralClustering(n_clusters=n_clusters, affinity='rbf', gamma=gamma, 
                        random_state=fixture['params']['randomState'])
labels = sc.fit_predict(X)
from sklearn.metrics import adjusted_rand_score
ari = adjusted_rand_score(fixture['labels'], labels)
print(f"   SpectralClustering ARI: {ari:.4f}")

# Get the affinity matrix that SpectralClustering actually used
print(f"\n7. Checking SpectralClustering's affinity matrix:")
print(f"   Affinity matrix from sc: shape={sc.affinity_matrix_.shape}")
print(f"   Matches our affinity? {np.allclose(sc.affinity_matrix_, affinity)}")