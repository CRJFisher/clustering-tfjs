import numpy as np
import json
from sklearn.cluster import SpectralClustering
from scipy.sparse.csgraph import laplacian
from sklearn.manifold import spectral_embedding
from sklearn.preprocessing import normalize

def load_fixture(filename):
    with open(f'test/fixtures/spectral/{filename}', 'r') as f:
        fixture = json.load(f)
    return fixture

def test_row_normalization():
    print("=== Testing if sklearn applies row normalization ===\n")
    
    # Test both moons and blobs
    datasets = ['moons_n2_knn.json', 'blobs_n2_knn.json']
    
    for dataset in datasets:
        print(f"\n--- {dataset} ---")
        fixture = load_fixture(dataset)
        X = np.array(fixture['X'])
        params = fixture['params']
        
        # Create SpectralClustering object
        sc = SpectralClustering(
            n_clusters=params['nClusters'],
            affinity='nearest_neighbors',
            n_neighbors=params['nNeighbors'],
            random_state=params['randomState']
        )
        
        # Fit to get affinity matrix
        sc.fit(X)
        affinity_matrix = sc.affinity_matrix_
        
        # Get sklearn's embedding used for clustering
        # Access the internal embedding after fit
        sklearn_embedding = sc.embedding_
        
        print(f"sklearn's internal embedding shape: {sklearn_embedding.shape}")
        
        # Check if it's row normalized
        row_norms = np.linalg.norm(sklearn_embedding, axis=1)
        print(f"Row norms - min: {row_norms.min():.6f}, max: {row_norms.max():.6f}")
        print(f"All rows normalized to 1.0: {np.allclose(row_norms, 1.0)}")
        
        # Get spectral_embedding function result
        raw_embedding = spectral_embedding(
            affinity_matrix,
            n_components=params['nClusters'],
            drop_first=False,
            random_state=params['randomState']
        )
        
        # Normalize it
        normalized_embedding = normalize(raw_embedding, norm='l2', axis=1)
        
        # Check if normalized version matches sklearn's internal embedding
        match = np.allclose(normalized_embedding, sklearn_embedding, atol=1e-6)
        print(f"Normalized spectral_embedding matches sc.embedding_: {match}")
        
        # Show first few rows
        print("\nFirst 3 rows:")
        print("  sklearn internal:", sklearn_embedding[:3])
        print("  normalized spectral_embedding:", normalized_embedding[:3])

if __name__ == "__main__":
    test_row_normalization()