import numpy as np
import json
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.manifold import spectral_embedding
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh

fixture_name = 'circles_n2_rbf'

print(f"Debugging full pipeline for {fixture_name}:\n")

with open(f'./test/fixtures/spectral/{fixture_name}.json') as f:
    data = json.load(f)

X = np.array(data['X'])
y_true = np.array(data['labels'])
n_clusters = len(np.unique(y_true))

# Create affinity
affinity = rbf_kernel(X, gamma=1.0)

print("1. Eigendecomposition check:")
L = laplacian(affinity, normed=True)
eigenvalues, eigenvectors = eigh(L)
print(f"   First eigenvalue: {eigenvalues[0]:.6e} (should be ~0)")
print(f"   Second eigenvalue: {eigenvalues[1]:.6f}")

# Check how many zero eigenvalues
zero_threshold_tests = [1e-2, 1e-5, 1e-8, 1e-10]
for tol in zero_threshold_tests:
    num_zeros = np.sum(eigenvalues <= tol)
    print(f"   Eigenvalues <= {tol}: {num_zeros}")

print("\n2. Spectral embedding (what sklearn uses for clustering):")
# This is what sklearn actually uses
embedding = spectral_embedding(
    affinity,
    n_components=n_clusters,
    drop_first=False,  # SpectralClustering uses False
    random_state=42
)

print(f"   Embedding shape: {embedding.shape}")
for i in range(embedding.shape[1]):
    vec = embedding[:, i]
    print(f"   Dimension {i}: unique values = {len(np.unique(np.round(vec, 8)))}, std = {vec.std():.6f}")

# Apply diffusion map scaling manually
print("\n3. Manual diffusion map scaling:")
# Select first n_clusters eigenvectors
selected_evecs = eigenvectors[:, :n_clusters]
selected_evals = eigenvalues[:n_clusters]

# Apply sklearn's scaling: multiply by sqrt(1 - eigenvalue)
scaling_factors = np.sqrt(np.maximum(0, 1 - selected_evals))
scaled_embedding = selected_evecs * scaling_factors

print(f"   Scaling factors: {scaling_factors}")
for i in range(n_clusters):
    vec = scaled_embedding[:, i]
    print(f"   Scaled dimension {i}: std = {vec.std():.6f}")

# Compare with sklearn embedding
print("\n4. Comparing manual scaling with sklearn:")
for i in range(n_clusters):
    diff = np.abs(scaled_embedding[:, i]) - np.abs(embedding[:, i])
    print(f"   Dimension {i} max diff: {np.max(np.abs(diff)):.6e}")

# Run k-means on different embeddings
print("\n5. K-means clustering results:")

# On sklearn embedding
km1 = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels1 = km1.fit_predict(embedding)
ari1 = adjusted_rand_score(y_true, labels1)
print(f"   sklearn embedding -> ARI = {ari1:.4f}")

# On manual scaled embedding
km2 = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels2 = km2.fit_predict(scaled_embedding)
ari2 = adjusted_rand_score(y_true, labels2)
print(f"   Manual scaled embedding -> ARI = {ari2:.4f}")

# On unscaled eigenvectors
km3 = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels3 = km3.fit_predict(selected_evecs)
ari3 = adjusted_rand_score(y_true, labels3)
print(f"   Unscaled eigenvectors -> ARI = {ari3:.4f}")

# Full sklearn SpectralClustering
sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
labels_sc = sc.fit_predict(affinity)
ari_sc = adjusted_rand_score(y_true, labels_sc)
print(f"   Full SpectralClustering -> ARI = {ari_sc:.4f}")

print("\n6. Key insights:")
if ari1 >= 0.99 and ari2 >= 0.99:
    print("   ✓ Both sklearn and manual scaling achieve perfect clustering")
    print("   → The issue is likely in our implementation details")
else:
    print("   ✗ Even manual scaling doesn't match sklearn perfectly")
    print("   → Need to check exact sklearn implementation")

# Check if the issue is the zero eigenvalue threshold
print("\n7. Impact of zero eigenvalue threshold:")
print(f"   Our threshold (1e-2) would include {np.sum(eigenvalues <= 1e-2)} eigenvectors")
print(f"   A smaller threshold (1e-8) would include {np.sum(eigenvalues <= 1e-8)} eigenvectors")
print("   This affects which eigenvectors are selected for clustering!")