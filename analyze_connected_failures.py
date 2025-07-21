import numpy as np
import json
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse import csr_matrix

print("Analyzing why connected graph tests fail:\n")

# Test each failing fixture
failing_fixtures = ['circles_n2_rbf', 'circles_n3_knn', 'circles_n3_rbf', 'moons_n2_rbf', 'moons_n3_rbf']

for fixture_name in failing_fixtures:
    print(f"\n{fixture_name}:")
    print("-" * 40)
    
    with open(f'./test/fixtures/spectral/{fixture_name}.json') as f:
        data = json.load(f)
    
    X = np.array(data['X'])
    y_true = np.array(data['labels'])
    n_clusters = len(np.unique(y_true))
    
    # Create affinity based on type
    if 'knn' in fixture_name:
        affinity = kneighbors_graph(X, n_neighbors=10, mode='connectivity', include_self=True)
        affinity = 0.5 * (affinity + affinity.T)
    else:  # rbf
        affinity = rbf_kernel(X, gamma=1.0)
        affinity_sparse = csr_matrix(affinity)
    
    # Run sklearn's spectral clustering
    sc = SpectralClustering(
        n_clusters=n_clusters, 
        affinity='precomputed',
        random_state=42
    )
    
    if 'rbf' in fixture_name:
        # For RBF, sklearn uses dense matrix
        y_pred = sc.fit_predict(affinity)
    else:
        y_pred = sc.fit_predict(affinity)
    
    ari = adjusted_rand_score(y_true, y_pred)
    print(f"  sklearn ARI: {ari:.4f}")
    
    # Check the spectral embedding quality
    from sklearn.manifold import spectral_embedding
    embedding = spectral_embedding(
        affinity_sparse if 'rbf' in fixture_name else affinity,
        n_components=n_clusters,
        drop_first=False,
        random_state=42
    )
    
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Unique values per dimension:")
    for i in range(min(3, embedding.shape[1])):
        unique = len(np.unique(np.round(embedding[:, i], 6)))
        print(f"    Dim {i}: {unique} unique values")
    
    # The key insight: For connected graphs, the eigenvectors should have 
    # smooth variation, not constant values
    if ari >= 0.99:
        print(f"  ✓ sklearn achieves near-perfect clustering")
    else:
        print(f"  ⚠ Even sklearn struggles with this dataset (ARI={ari:.4f})")

print("\n\nConclusion:")
print("=" * 60)
print("For CONNECTED graphs, the issue is NOT about component indicators.")
print("The eigenvectors should have smooth variation to capture cluster structure.")
print("\nPossible reasons for our failures:")
print("1. Numerical accuracy of our eigendecomposition")
print("2. Different eigenvector ordering or sign flipping") 
print("3. Incorrect diffusion map scaling")
print("4. K-means initialization differences")