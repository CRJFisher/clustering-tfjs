import numpy as np
import json
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.manifold import spectral_embedding
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

fixture_name = 'circles_n2_rbf'

with open(f'./test/fixtures/spectral/{fixture_name}.json') as f:
    data = json.load(f)

X = np.array(data['X'])
y_true = np.array(data['labels'])
affinity = rbf_kernel(X, gamma=1.0)

print("Final check - what makes sklearn work?\n")

# 1. Get sklearn's full spectral clustering
sc = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=42)
labels_sklearn = sc.fit_predict(affinity)
ari_sklearn = adjusted_rand_score(y_true, labels_sklearn)
print(f"1. sklearn SpectralClustering ARI: {ari_sklearn:.4f}")

# 2. Get sklearn's embedding
embedding = spectral_embedding(affinity, n_components=2, drop_first=False, random_state=42)
print(f"\n2. sklearn embedding:")
print(f"   Shape: {embedding.shape}")
print(f"   Dim 0: constant = {embedding[0, 0]:.10f}")
print(f"   Dim 1: range = [{embedding[:, 1].min():.6f}, {embedding[:, 1].max():.6f}]")

# 3. Run k-means on sklearn embedding
from sklearn.cluster import KMeans
km = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_km = km.fit_predict(embedding)
ari_km = adjusted_rand_score(y_true, labels_km)
print(f"\n3. K-means on sklearn embedding: ARI = {ari_km:.4f}")

# 4. What if we DON'T use the constant first dimension?
embedding_no_const = embedding[:, 1:]  # Skip first dimension
km2 = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_km2 = km2.fit_predict(embedding_no_const)
ari_km2 = adjusted_rand_score(y_true, labels_km2)
print(f"\n4. K-means on dimension 1 only: ARI = {ari_km2:.4f}")

# 5. The key insight
print("\n5. KEY INSIGHT:")
if ari_km2 < 0.9:
    print("   The constant first dimension IS important for clustering!")
    print("   It provides separation that the second dimension alone doesn't have.")
else:
    print("   The second dimension alone gives good clustering.")
    
# 6. Let's check what happens with drop_first=True
embedding_drop = spectral_embedding(affinity, n_components=2, drop_first=True, random_state=42)
km3 = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_km3 = km3.fit_predict(embedding_drop)
ari_km3 = adjusted_rand_score(y_true, labels_km3)
print(f"\n6. With drop_first=True: ARI = {ari_km3:.4f}")

print("\n7. Summary:")
print(f"   Full SpectralClustering: {ari_sklearn:.4f}")
print(f"   Manual k-means on full embedding: {ari_km:.4f}")
print(f"   Manual k-means on dim 1 only: {ari_km2:.4f}")
print(f"   Manual k-means with drop_first=True: {ari_km3:.4f}")