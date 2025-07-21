import numpy as np
import json
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.manifold import spectral_embedding
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh

# Test on circles_n2_rbf which has ARI=0.869
fixture_name = 'circles_n2_rbf'

with open(f'./test/fixtures/spectral/{fixture_name}.json') as f:
    data = json.load(f)

X = np.array(data['X'])
y_true = np.array(data['labels'])
affinity = rbf_kernel(X, gamma=1.0)

print(f"Analyzing k-means sensitivity for {fixture_name}:\n")

# Get sklearn's embedding
embedding_sk = spectral_embedding(affinity, n_components=2, drop_first=False, random_state=42)

# Get our embedding (simulated with small perturbations)
L = laplacian(affinity, normed=True)
eigenvalues, eigenvectors = eigh(L)
our_embedding = np.column_stack([
    np.ones(len(X)) / np.sqrt(len(X)),  # constant eigenvector
    eigenvectors[:, 1] * np.sqrt(1 - eigenvalues[1])  # second eigenvector scaled
])

# Add small perturbations to simulate our numerical differences
np.random.seed(42)
perturbation = np.random.normal(0, 0.001, our_embedding.shape)
our_embedding_perturbed = our_embedding + perturbation

print("1. Effect of different random seeds on sklearn embedding:")
aris_sklearn = []
for seed in range(10):
    km = KMeans(n_clusters=2, random_state=seed, n_init=1)
    labels = km.fit_predict(embedding_sk)
    ari = adjusted_rand_score(y_true, labels)
    aris_sklearn.append(ari)
    print(f"   Seed {seed}: ARI = {ari:.4f}")

print(f"\n   Mean ARI: {np.mean(aris_sklearn):.4f}")
print(f"   Std ARI: {np.std(aris_sklearn):.4f}")

print("\n2. Effect of different random seeds on perturbed embedding:")
aris_perturbed = []
for seed in range(10):
    km = KMeans(n_clusters=2, random_state=seed, n_init=1)
    labels = km.fit_predict(our_embedding_perturbed)
    ari = adjusted_rand_score(y_true, labels)
    aris_perturbed.append(ari)
    print(f"   Seed {seed}: ARI = {ari:.4f}")

print(f"\n   Mean ARI: {np.mean(aris_perturbed):.4f}")
print(f"   Std ARI: {np.std(aris_perturbed):.4f}")

print("\n3. Effect of n_init parameter:")
for n_init in [1, 10, 50, 100]:
    km = KMeans(n_clusters=2, random_state=42, n_init=n_init)
    labels = km.fit_predict(our_embedding_perturbed)
    ari = adjusted_rand_score(y_true, labels)
    print(f"   n_init={n_init}: ARI = {ari:.4f}")

print("\n4. Consensus clustering approach:")
# Run k-means multiple times and take majority vote
n_runs = 50
all_labels = []
for i in range(n_runs):
    km = KMeans(n_clusters=2, random_state=i, n_init=1)
    labels = km.fit_predict(our_embedding_perturbed)
    all_labels.append(labels)

# Convert to matrix for easier voting
all_labels = np.array(all_labels)

# For each point, take the most common label
consensus_labels = []
for i in range(len(X)):
    # Handle label switching by checking which assignment is more common
    votes = all_labels[:, i]
    unique, counts = np.unique(votes, return_counts=True)
    consensus_labels.append(unique[np.argmax(counts)])

consensus_ari = adjusted_rand_score(y_true, consensus_labels)
print(f"\nConsensus clustering ARI: {consensus_ari:.4f}")

print("\n5. Analyzing cluster stability:")
# Check how often points switch clusters across runs
switching_frequency = []
for i in range(len(X)):
    labels_for_point = all_labels[:, i]
    switches = np.sum(np.diff(labels_for_point) != 0)
    switching_frequency.append(switches / (n_runs - 1))

print(f"   Points that never switch: {np.sum(np.array(switching_frequency) == 0)}")
print(f"   Points that switch >50% of time: {np.sum(np.array(switching_frequency) > 0.5)}")
print(f"   Mean switching frequency: {np.mean(switching_frequency):.4f}")