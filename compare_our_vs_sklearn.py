import numpy as np
import json
import subprocess
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

# Test one failing case
fixture_name = 'circles_n2_rbf'

print(f"Comparing our implementation vs sklearn for {fixture_name}:\n")

with open(f'./test/fixtures/spectral/{fixture_name}.json') as f:
    data = json.load(f)

X = np.array(data['X'])
y_true = np.array(data['labels'])

# Run our implementation
result = subprocess.run(
    ['npm', 'test', '--', '--testPathPattern=spectral', '--testNamePattern=circles_n2_rbf'],
    capture_output=True,
    text=True
)

# Extract ARI from our test output
if 'expected' in result.stdout:
    lines = result.stdout.split('\n')
    for line in lines:
        if 'expected' in line and 'toBeCloseTo' in line:
            # Extract the values
            import re
            match = re.search(r'Expected: ([\d.]+).*?Received: ([\d.]+)', line)
            if match:
                expected_ari = float(match.group(1))
                our_ari = float(match.group(2))
                print(f"Our ARI: {our_ari:.4f}")
                print(f"Expected (sklearn) ARI: {expected_ari:.4f}")
                print(f"Difference: {abs(expected_ari - our_ari):.4f}")

# Let's check intermediate values
print("\nDiagnosing the issue:")

# 1. Check affinity matrix
affinity = rbf_kernel(X, gamma=1.0)
print(f"\n1. Affinity matrix:")
print(f"   Shape: {affinity.shape}")
print(f"   Min: {affinity.min():.6f}, Max: {affinity.max():.6f}")
print(f"   Sparsity (< 1e-10): {(affinity < 1e-10).sum() / affinity.size:.2%}")

# 2. Check spectral embedding
from sklearn.manifold import spectral_embedding
embedding = spectral_embedding(affinity, n_components=2, drop_first=False, random_state=42)

print(f"\n2. sklearn embedding:")
for i in range(2):
    vec = embedding[:, i]
    print(f"   Eigenvector {i}: min={vec.min():.6f}, max={vec.max():.6f}, std={vec.std():.6f}")

# 3. Run k-means on sklearn embedding
from sklearn.cluster import KMeans
km = KMeans(n_clusters=2, random_state=42, n_init=10)
sklearn_labels = km.fit_predict(embedding)
sklearn_ari = adjusted_rand_score(y_true, sklearn_labels)
print(f"\n3. K-means on sklearn embedding: ARI = {sklearn_ari:.4f}")

# 4. Check if issue is in eigenvector computation or k-means
print("\n4. Hypothesis testing:")
print("   If sklearn k-means on sklearn embedding gives ARI=1.0,")
print("   then our issue is likely in eigenvector computation.")
print("   If it gives lower ARI, then k-means initialization might differ.")

# Additional check: eigenvalue magnitudes
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh

L = laplacian(affinity, normed=True)
eigenvalues, _ = eigh(L)
print(f"\n5. Eigenvalue spectrum (first 5): {eigenvalues[:5]}")
print(f"   These should be very close to 0 for good cluster separation")