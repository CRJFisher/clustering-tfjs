import numpy as np
import json
from sklearn.cluster import SpectralClustering
from sklearn.manifold import spectral_embedding
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse.csgraph import laplacian

# Load the same test data
with open('test/fixtures/spectral/circles_n2_rbf.json', 'r') as f:
    data = json.load(f)

X = np.array(data['X'])
params = data['params']

print("Input shape:", X.shape)

# Compute affinity matrix
gamma = params.get('gamma', 1.0)
affinity = rbf_kernel(X, gamma=gamma)
print("\nAffinity matrix:")
print("  Shape:", affinity.shape)
print("  Min:", affinity.min())
print("  Max:", affinity.max())
print("  Sum:", affinity.sum())

# Save affinity for comparison
import os
os.makedirs('tmp', exist_ok=True)
np.save('tmp/sklearn_affinity.npy', affinity)

# Compute Laplacian
lap, dd = laplacian(affinity, normed=True, return_diag=True)
print("\nLaplacian:")
print("  Shape:", lap.shape)
print("  Min:", lap.min())
print("  Max:", lap.max())
print("  Diagonal:", np.diag(lap)[:5], "...")

# Get spectral embedding
embedding = spectral_embedding(
    affinity,
    n_components=params['nClusters'],
    drop_first=False,
    random_state=params['randomState']
)
print("\nSpectral embedding:")
print("  Shape:", embedding.shape)
print("  First 5 rows:")
print(embedding[:5])

# Check if first column is constant
print("\n  Column statistics:")
for i in range(embedding.shape[1]):
    col = embedding[:, i]
    print(f"    Column {i}: min={col.min():.6f}, max={col.max():.6f}, std={col.std():.6f}")

# Save embedding for comparison
np.save('tmp/sklearn_embedding.npy', embedding)

# Run full spectral clustering
model = SpectralClustering(
    n_clusters=params['nClusters'],
    affinity='precomputed',
    random_state=params['randomState'],
    assign_labels='kmeans'
)
labels = model.fit_predict(affinity)

print("\nFinal labels:", labels)
print("Label counts:", {i: (labels == i).sum() for i in range(params['nClusters'])})