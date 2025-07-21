import numpy as np
import json
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.manifold import spectral_embedding
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh

fixture_name = 'circles_n2_rbf'

with open(f'./test/fixtures/spectral/{fixture_name}.json') as f:
    data = json.load(f)

X = np.array(data['X'])
y_true = np.array(data['labels'])
affinity = rbf_kernel(X, gamma=1.0)

print("Comparing dimensions in detail:\n")

# Get sklearn embedding
embedding = spectral_embedding(affinity, n_components=2, drop_first=False, random_state=42)

# Get raw eigenvectors
L = laplacian(affinity, normed=True)
eigenvalues, eigenvectors = eigh(L)

# Our second dimension (eigenvector 1 scaled)
our_dim1 = eigenvectors[:, 1] * np.sqrt(1 - eigenvalues[1])

print("1. Range comparison:")
print(f"   sklearn dim 1: [{embedding[:, 1].min():.6f}, {embedding[:, 1].max():.6f}]")
print(f"   Our dim 1: [{our_dim1.min():.6f}, {our_dim1.max():.6f}]")

# Check correlation
corr = np.corrcoef(embedding[:, 1], our_dim1)[0, 1]
print(f"\n2. Correlation: {corr:.6f}")

# Check if it's just a sign flip
corr_neg = np.corrcoef(embedding[:, 1], -our_dim1)[0, 1]
print(f"   Correlation with negated: {corr_neg:.6f}")

# The actual vectors
if abs(corr_neg) > abs(corr):
    our_dim1_aligned = -our_dim1
else:
    our_dim1_aligned = our_dim1

# Check scaling
ratio = embedding[:, 1][0] / our_dim1_aligned[0]
print(f"\n3. Scaling ratio: {ratio:.6f}")

# Apply the scaling
our_dim1_rescaled = our_dim1_aligned * ratio

# Check if they match now
diff = np.abs(embedding[:, 1] - our_dim1_rescaled)
print(f"\n4. After scaling:")
print(f"   Max difference: {diff.max():.10f}")
print(f"   Mean difference: {diff.mean():.10f}")

# The key test: k-means on our rescaled dimension
print("\n5. K-means results:")
km1 = KMeans(n_clusters=2, random_state=42, n_init=10)
labels1 = km1.fit_predict(embedding[:, 1].reshape(-1, 1))
ari1 = adjusted_rand_score(y_true, labels1)
print(f"   sklearn dim 1: ARI = {ari1:.4f}")

km2 = KMeans(n_clusters=2, random_state=42, n_init=10)
labels2 = km2.fit_predict(our_dim1_rescaled.reshape(-1, 1))
ari2 = adjusted_rand_score(y_true, labels2)
print(f"   Our rescaled dim 1: ARI = {ari2:.4f}")

# Check the full embedding approach
print("\n6. Full embedding k-means:")
our_const = np.ones(len(X)) / np.sqrt(len(X))
our_embedding = np.column_stack([our_const, our_dim1_aligned])
our_embedding_scaled = our_embedding * ratio

km3 = KMeans(n_clusters=2, random_state=42, n_init=10)
labels3 = km3.fit_predict(our_embedding_scaled)
ari3 = adjusted_rand_score(y_true, labels3)
print(f"   Our full embedding (scaled): ARI = {ari3:.4f}")