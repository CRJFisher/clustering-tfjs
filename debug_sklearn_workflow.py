import numpy as np
import json
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.cluster._spectral import spectral_embedding
from sklearn.metrics import adjusted_rand_score
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh

# Load fixture
with open('./test/fixtures/spectral/circles_n2_rbf.json', 'r') as f:
    fixture = json.load(f)

X = np.array(fixture['X'])
gamma = fixture['params']['gamma']
n_clusters = fixture['params']['nClusters']
random_state = fixture['params']['randomState']
expected_labels = fixture['labels']

print("Debugging sklearn spectral clustering workflow")
print("=" * 60)

# Step 1: Compute affinity
affinity = rbf_kernel(X, gamma=gamma)
print(f"\n1. Affinity matrix computed")

# Step 2: Get spectral embedding
embedding = spectral_embedding(
    affinity, 
    n_components=n_clusters,
    drop_first=False,  # Important: spectral clustering uses drop_first=False
    eigen_solver='arpack'
)
print(f"\n2. Spectral embedding shape: {embedding.shape}")
print(f"   First row of embedding: {embedding[0]}")
print(f"   Embedding range: [{embedding.min():.6f}, {embedding.max():.6f}]")

# Step 3: Run k-means on embedding
km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
labels = km.fit_predict(embedding)
ari = adjusted_rand_score(expected_labels, labels)
print(f"\n3. K-means clustering:")
print(f"   ARI: {ari:.4f}")

# Let's also manually compute the embedding to see each step
print("\n4. Manual computation:")

# Normalized Laplacian
laplacian, dd = csgraph.laplacian(affinity, normed=True, return_diag=True)

# Eigendecomposition
eigenvalues, eigenvectors = eigsh(laplacian, k=n_clusters, which='SM')
print(f"   Eigenvalues: {eigenvalues}")

# Diffusion map scaling (this is what spectral_embedding does)
# For normalized Laplacian: scale by sqrt(1 - eigenvalue)
scaling = np.sqrt(np.maximum(0, 1 - eigenvalues))
embedding_manual = eigenvectors * scaling
print(f"   Scaling factors: {scaling}")
print(f"   Manual embedding first row: {embedding_manual[0]}")

# Check if embeddings match
embedding_diff = np.abs(embedding - embedding_manual).max()
print(f"   Max difference between sklearn and manual: {embedding_diff:.6f}")

# Try k-means with different seeds
print("\n5. K-means sensitivity test:")
for seed in range(5):
    km_test = KMeans(n_clusters=n_clusters, random_state=seed, n_init=1)
    labels_test = km_test.fit_predict(embedding)
    ari_test = adjusted_rand_score(expected_labels, labels_test)
    print(f"   Seed {seed}: ARI = {ari_test:.4f}")

# Save embedding for comparison
results = {
    'embedding': embedding.tolist(),
    'eigenvalues': eigenvalues.tolist(),
    'eigenvectors_raw': eigenvectors.tolist(),
    'scaling_factors': scaling.tolist(),
    'sklearn_ari': float(ari)
}

with open('sklearn_workflow_debug.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to sklearn_workflow_debug.json")