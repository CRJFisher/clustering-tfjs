import numpy as np
import json
from sklearn.cluster import SpectralClustering
from scipy.sparse.csgraph import laplacian
# from sklearn.manifold._spectral_embedding import _spectral_embedding
from sklearn.manifold import spectral_embedding

def debug_eigenvector_selection():
    print("=== Debugging sklearn's eigenvector selection ===\n")
    
    # Load moons dataset
    with open('test/fixtures/spectral/moons_n2_knn.json', 'r') as f:
        fixture = json.load(f)
    
    X = np.array(fixture['X'])
    params = fixture['params']
    
    # Create SpectralClustering
    sc = SpectralClustering(
        n_clusters=params['nClusters'],
        affinity='nearest_neighbors',
        n_neighbors=params['nNeighbors'],
        random_state=params['randomState']
    )
    
    # Fit to get affinity matrix
    sc.fit(X)
    affinity_matrix = sc.affinity_matrix_
    
    # Get Laplacian
    L, diag = laplacian(affinity_matrix, normed=True, return_diag=True)
    if hasattr(L, 'todense'):
        L_dense = L.todense()
    else:
        L_dense = L
    
    # Get all eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
    
    print(f"First 10 eigenvalues:")
    print(eigenvalues[:10])
    
    # Check what spectral_embedding returns
    print("\n--- Using spectral_embedding with drop_first=False ---")
    embedding_no_drop = spectral_embedding(
        affinity_matrix,
        n_components=params['nClusters'],
        drop_first=False,
        random_state=params['randomState']
    )
    
    print(f"Embedding shape: {embedding_no_drop.shape}")
    print(f"First column unique values: {len(np.unique(np.round(embedding_no_drop[:, 0], 6)))}")
    print(f"First column std: {np.std(embedding_no_drop[:, 0]):.10f}")
    print(f"First 5 values of first column: {embedding_no_drop[:5, 0]}")
    
    print("\n--- Using spectral_embedding with drop_first=True ---")
    embedding_drop = spectral_embedding(
        affinity_matrix,
        n_components=params['nClusters'],
        drop_first=True,
        random_state=params['randomState']
    )
    
    print(f"Embedding shape: {embedding_drop.shape}")
    print(f"First column unique values: {len(np.unique(np.round(embedding_drop[:, 0], 6)))}")
    print(f"First 5 values of first column: {embedding_drop[:5, 0]}")
    
    # Check if embeddings match eigenvectors
    print("\n--- Checking which eigenvectors are used ---")
    
    # First eigenvector (constant one)
    first_eigvec = eigenvectors[:, 0]
    print(f"\nEigenvector 0 (eigenvalue={eigenvalues[0]:.6f}):")
    print(f"  Unique values: {len(np.unique(np.round(first_eigvec, 6)))}")
    print(f"  Std: {np.std(first_eigvec):.10f}")
    
    # Apply diffusion scaling to first eigenvector
    scaled_first = first_eigvec * np.sqrt(max(0, 1 - eigenvalues[0]))
    print(f"  After diffusion scaling: unique={len(np.unique(np.round(scaled_first, 6)))}")
    
    # Check if it matches embedding
    if np.allclose(np.abs(embedding_no_drop[:, 0]), np.abs(scaled_first), atol=1e-4):
        print("  ✓ Matches first column of drop_first=False embedding")
    
    # Check subsequent eigenvectors
    for i in range(1, 3):
        eigvec = eigenvectors[:, i]
        scaled = eigvec * np.sqrt(max(0, 1 - eigenvalues[i]))
        
        print(f"\nEigenvector {i} (eigenvalue={eigenvalues[i]:.6f}):")
        print(f"  Unique values: {len(np.unique(np.round(eigvec, 6)))}")
        
        # Check against drop_first=False embedding
        if i < embedding_no_drop.shape[1]:
            if np.allclose(np.abs(embedding_no_drop[:, i]), np.abs(scaled), atol=1e-4):
                print(f"  ✓ Matches column {i} of drop_first=False embedding")
                
        # Check against drop_first=True embedding  
        if i-1 < embedding_drop.shape[1]:
            if np.allclose(np.abs(embedding_drop[:, i-1]), np.abs(scaled), atol=1e-4):
                print(f"  ✓ Matches column {i-1} of drop_first=True embedding")

if __name__ == "__main__":
    debug_eigenvector_selection()