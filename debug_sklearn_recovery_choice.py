import numpy as np
import json
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
from scipy.sparse.csgraph import laplacian
from sklearn.manifold import spectral_embedding

def load_fixture(filename):
    with open(f'test/fixtures/spectral/{filename}', 'r') as f:
        fixture = json.load(f)
    return fixture

def test_sklearn_recovery_behavior():
    print("=== Testing sklearn's eigenvector recovery behavior ===\n")
    
    # Test all datasets to see pattern
    datasets = [
        'blobs_n2_knn.json',     # Disconnected - we know this uses recovery
        'blobs_n2_rbf.json',     # Should be connected
        'moons_n2_knn.json',     # Connected - seems NOT to use recovery
        'moons_n2_rbf.json',     # Connected - seems NOT to use recovery
    ]
    
    for dataset_name in datasets:
        print(f"\n--- {dataset_name} ---")
        fixture = load_fixture(dataset_name)
        X = np.array(fixture['X'])
        y_true = np.array(fixture['labels'])
        params = fixture['params']
        
        # Create SpectralClustering object
        if 'knn' in dataset_name:
            sc = SpectralClustering(
                n_clusters=params['nClusters'],
                affinity='nearest_neighbors',
                n_neighbors=params['nNeighbors'],
                random_state=params['randomState']
            )
        else:
            sc = SpectralClustering(
                n_clusters=params['nClusters'],
                affinity='rbf',
                gamma=params['gamma'],
                random_state=params['randomState']
            )
        
        # Fit and get predictions
        y_pred = sc.fit_predict(X)
        ari = adjusted_rand_score(y_true, y_pred)
        print(f"ARI: {ari:.4f}")
        
        # Get affinity matrix
        affinity_matrix = sc.affinity_matrix_
        
        # Compute normalized Laplacian with diag
        L, diag = laplacian(affinity_matrix, normed=True, return_diag=True)
        
        # Convert to dense if sparse
        if hasattr(L, 'todense'):
            L_dense = L.todense()
        else:
            L_dense = L
        
        # Get eigenvalues to check connectivity
        eigenvalues = np.sort(np.linalg.eigvalsh(L_dense))[:5]
        n_components = np.sum(eigenvalues < 1e-8)
        print(f"Number of connected components: {n_components}")
        print(f"First 5 eigenvalues: {eigenvalues}")
        
        # Get spectral embedding
        embedding = spectral_embedding(
            affinity_matrix,
            n_components=params['nClusters'],
            drop_first=False,
            eigen_solver='arpack',
            random_state=params['randomState']
        )
        
        # Get eigenvectors directly
        eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
        idx = eigenvalues.argsort()
        eigenvectors = eigenvectors[:, idx[:params['nClusters']]]
        
        # Apply recovery
        sqrt_diag = np.sqrt(diag)
        eigenvectors_recovered = eigenvectors / sqrt_diag[:, np.newaxis]
        
        # Check which one matches sklearn's embedding
        print("\nUnique values in first eigenvector/embedding:")
        print(f"  Raw eigenvector[0]: {len(np.unique(np.round(eigenvectors[:, 0], 6)))} unique values")
        print(f"  Recovered eigenvector[0]: {len(np.unique(np.round(eigenvectors_recovered[:, 0], 6)))} unique values")
        print(f"  sklearn embedding[0]: {len(np.unique(np.round(embedding[:, 0], 6)))} unique values")
        
        # Check if embedding matches raw or recovered
        raw_match = np.allclose(np.abs(embedding[:, 0]), np.abs(eigenvectors[:, 0]), atol=1e-4)
        recovered_match = np.allclose(np.abs(embedding[:, 0]), np.abs(eigenvectors_recovered[:, 0]), atol=1e-4)
        
        print(f"\nDoes sklearn use recovery?")
        print(f"  Matches raw eigenvectors: {raw_match}")
        print(f"  Matches recovered eigenvectors: {recovered_match}")
        
        # Additional check: is first eigenvector constant?
        is_constant = np.std(embedding[:, 0]) < 1e-10
        print(f"  First embedding dimension is constant: {is_constant}")

if __name__ == "__main__":
    test_sklearn_recovery_behavior()