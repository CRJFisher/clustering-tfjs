import numpy as np
import json
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.manifold import spectral_embedding

# Load fixture
with open('./test/fixtures/spectral/circles_n3_rbf.json', 'r') as f:
    fixture = json.load(f)

X = np.array(fixture['X'])
n_clusters = fixture['params']['nClusters']
gamma = fixture['params']['gamma']
random_state = fixture['params']['randomState']

print("Testing k-means seed sensitivity")
print("=" * 60)

# Get the embedding once
affinity = rbf_kernel(X, gamma=gamma)
embedding = spectral_embedding(affinity, n_components=n_clusters, drop_first=False)

# Test multiple seeds
from sklearn.cluster import KMeans

print("\nTesting different random seeds on the same embedding:")
best_ari = 0
best_seed = None
best_labels = None

for seed in range(100):
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=1)
    labels = km.fit_predict(embedding)
    ari = adjusted_rand_score(fixture['labels'], labels)
    
    if ari > best_ari:
        best_ari = ari
        best_seed = seed
        best_labels = labels
    
    if seed < 10 or ari > 0.9:
        print(f"  Seed {seed}: ARI = {ari:.4f}")

print(f"\nBest result: seed={best_seed}, ARI={best_ari:.4f}")

# Now test SpectralClustering with different seeds
print("\nTesting SpectralClustering with different random_state values:")
for seed in [0, 1, 10, 42, 100]:
    sc = SpectralClustering(
        n_clusters=n_clusters, 
        affinity='rbf', 
        gamma=gamma, 
        random_state=seed,
        n_init=10
    )
    labels = sc.fit_predict(X)
    ari = adjusted_rand_score(fixture['labels'], labels)
    print(f"  random_state={seed}: ARI = {ari:.4f}")

# Check what happens with our original seed but different n_init values
print(f"\nOriginal random_state={random_state} with different n_init:")
for n_init in [1, 5, 10, 20, 50]:
    sc = SpectralClustering(
        n_clusters=n_clusters, 
        affinity='rbf', 
        gamma=gamma, 
        random_state=random_state,
        n_init=n_init
    )
    labels = sc.fit_predict(X)
    ari = adjusted_rand_score(fixture['labels'], labels)
    print(f"  n_init={n_init}: ARI = {ari:.4f}")