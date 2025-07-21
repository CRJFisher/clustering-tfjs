import numpy as np
import json
from sklearn.cluster import SpectralClustering
from scipy.sparse.csgraph import laplacian
from sklearn.manifold import spectral_embedding

def load_fixture(filename):
    with open(f'test/fixtures/spectral/{filename}', 'r') as f:
        fixture = json.load(f)
    return fixture

def verify_diffusion_scaling():
    print("=== Verifying sklearn's diffusion map scaling ===\n")
    
    # Test moons dataset where we see the issue
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
    
    # Compute normalized Laplacian
    L, diag = laplacian(affinity_matrix, normed=True, return_diag=True)
    
    # Convert to dense
    if hasattr(L, 'todense'):
        L_dense = L.todense()
    else:
        L_dense = L
    
    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
    idx = eigenvalues.argsort()
    n_components = params['nClusters']
    eigenvalues = eigenvalues[idx[:n_components]]
    eigenvectors = eigenvectors[:, idx[:n_components]]
    
    print(f"First {n_components} eigenvalues: {eigenvalues}")
    
    # Get sklearn's embedding
    embedding = spectral_embedding(
        affinity_matrix,
        n_components=n_components,
        drop_first=False,
        random_state=params['randomState']
    )
    
    print("\nComparing eigenvectors with sklearn's embedding:")
    
    # Test different scalings
    for i in range(n_components):
        print(f"\nEigenvector {i} (eigenvalue = {eigenvalues[i]:.6f}):")
        
        # Raw eigenvector
        raw = eigenvectors[:, i]
        
        # Diffusion scaling: scale by sqrt(1 - eigenvalue)
        diffusion_scale = np.sqrt(max(0, 1 - eigenvalues[i]))
        diffusion_scaled = raw * diffusion_scale
        
        # sklearn's embedding
        sklearn_emb = embedding[:, i]
        
        # Compare norms
        print(f"  Raw eigenvector norm: {np.linalg.norm(raw):.4f}")
        print(f"  Diffusion scaled norm: {np.linalg.norm(diffusion_scaled):.4f}")
        print(f"  sklearn embedding norm: {np.linalg.norm(sklearn_emb):.4f}")
        
        # Check if they match (up to sign)
        match_raw = np.allclose(np.abs(sklearn_emb), np.abs(raw), atol=1e-3)
        match_diffusion = np.allclose(np.abs(sklearn_emb), np.abs(diffusion_scaled), atol=1e-3)
        
        print(f"  Matches raw: {match_raw}")
        print(f"  Matches diffusion scaled: {match_diffusion}")
        
        # Check unique values
        print(f"  Unique values - raw: {len(np.unique(np.round(raw, 6)))}, "
              f"sklearn: {len(np.unique(np.round(sklearn_emb, 6)))}")
        
        # For constant eigenvector, check the actual value
        if np.std(raw) < 1e-10:
            print(f"  Constant eigenvector value: raw = {raw[0]:.6f}, "
                  f"sklearn = {sklearn_emb[0]:.6f}")
            print(f"  Expected diffusion scaled value: {raw[0] * diffusion_scale:.6f}")

if __name__ == "__main__":
    verify_diffusion_scaling()