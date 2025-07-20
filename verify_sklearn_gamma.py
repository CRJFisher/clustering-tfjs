#!/usr/bin/env python3
import json
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import adjusted_rand_score

# Load the fixture
with open('test/fixtures/spectral/blobs_n2_rbf.json', 'r') as f:
    fixture = json.load(f)

X = np.array(fixture['X'])
expected_labels = np.array(fixture['labels'])
gamma = fixture['params']['gamma']

print(f"Dataset shape: {X.shape}")
print(f"Gamma from fixture: {gamma}")
print(f"Default gamma (1/n_features): {1.0 / X.shape[1]}")

# Compute RBF affinity with fixture gamma
affinity_fixture = rbf_kernel(X, gamma=gamma)
print(f"\nWith gamma={gamma}:")
print(f"Min affinity (off-diag): {np.min(affinity_fixture[affinity_fixture < 1.0]):.6f}")
print(f"Max affinity (off-diag): {np.max(affinity_fixture[affinity_fixture < 1.0]):.6f}")
print(f"Mean affinity (off-diag): {np.mean(affinity_fixture[affinity_fixture < 1.0]):.6f}")

# Try spectral clustering with this gamma
model1 = SpectralClustering(n_clusters=2, affinity='rbf', gamma=gamma, random_state=42)
labels1 = model1.fit_predict(X)
ari1 = adjusted_rand_score(expected_labels, labels1)
print(f"ARI with gamma={gamma}: {ari1:.4f}")

# Try with default gamma
model2 = SpectralClustering(n_clusters=2, affinity='rbf', gamma=None, random_state=42)
labels2 = model2.fit_predict(X)
ari2 = adjusted_rand_score(expected_labels, labels2)
print(f"\nARI with default gamma: {ari2:.4f}")

# Try with smaller gamma values
for test_gamma in [0.1, 0.01]:
    model = SpectralClustering(n_clusters=2, affinity='rbf', gamma=test_gamma, random_state=42)
    labels = model.fit_predict(X)
    ari = adjusted_rand_score(expected_labels, labels)
    print(f"ARI with gamma={test_gamma}: {ari:.4f}")