import numpy as np
import json
from sklearn.cluster import SpectralClustering
from scipy.sparse.csgraph import laplacian
from sklearn.manifold import spectral_embedding

def load_fixture(filename):
    with open(f'test/fixtures/spectral/{filename}', 'r') as f:
        fixture = json.load(f)
    return fixture

def debug_sklearn_normalization():
    print("=== Debugging sklearn's eigenvector normalization ===\n")
    
    # Test moons dataset
    dataset = 'moons_n2_knn.json'
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
    
    # Get sklearn's embedding
    embedding = spectral_embedding(
        affinity_matrix,
        n_components=params['nClusters'],
        drop_first=False,
        random_state=params['randomState']
    )
    
    print(f"Embedding shape: {embedding.shape}")
    print(f"Number of samples: {X.shape[0]}")
    
    # Check row-wise normalization
    print("\nRow-wise norms of embedding:")
    row_norms = np.linalg.norm(embedding, axis=1)
    print(f"  Min: {row_norms.min():.6f}")
    print(f"  Max: {row_norms.max():.6f}")
    print(f"  Mean: {row_norms.mean():.6f}")
    print(f"  Std: {row_norms.std():.6f}")
    
    # Check if all rows have same norm
    all_same = np.allclose(row_norms, row_norms[0], atol=1e-6)
    print(f"  All rows have same norm: {all_same}")
    
    # Check specific values
    print("\nFirst 5 rows of embedding:")
    print(embedding[:5])
    
    # Check column norms
    print("\nColumn norms:")
    for i in range(embedding.shape[1]):
        col_norm = np.linalg.norm(embedding[:, i])
        unique_vals = len(np.unique(np.round(embedding[:, i], 6)))
        print(f"  Column {i}: norm = {col_norm:.4f}, unique values = {unique_vals}")
    
    # Check if embedding is normalized to unit norm
    expected_norm = 1.0 / np.sqrt(X.shape[0])  # Unit vector in high-D space
    print(f"\nExpected norm if unit normalized: {expected_norm:.6f}")
    print(f"Actual first row norm: {row_norms[0]:.6f}")
    print(f"Match: {np.isclose(row_norms[0], expected_norm)}")

if __name__ == "__main__":
    debug_sklearn_normalization()