import numpy as np
import json
from sklearn.cluster import SpectralClustering
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh

def compare_eigensolvers():
    print("=== Comparing eigensolvers on moons dataset ===\n")
    
    # Load moons dataset
    with open('test/fixtures/spectral/moons_n2_knn.json', 'r') as f:
        fixture = json.load(f)
    
    X = np.array(fixture['X'])
    params = fixture['params']
    
    # Create affinity matrix
    sc = SpectralClustering(
        n_clusters=params['nClusters'],
        affinity='nearest_neighbors',
        n_neighbors=params['nNeighbors'],
        random_state=params['randomState']
    )
    sc.fit(X)
    affinity_matrix = sc.affinity_matrix_
    
    # Get Laplacian
    L, diag = laplacian(affinity_matrix, normed=True, return_diag=True)
    if hasattr(L, 'todense'):
        L_dense = L.todense()
        L_sparse = L
    else:
        L_dense = L
        from scipy.sparse import csr_matrix
        L_sparse = csr_matrix(L)
    
    print("1. Using scipy.linalg.eigh (dense):")
    eigenvalues_dense, eigenvectors_dense = eigh(L_dense)
    print(f"   First 5 eigenvalues: {eigenvalues_dense[:5]}")
    print(f"   First eigenvector unique values: {len(np.unique(np.round(eigenvectors_dense[:, 0], 6)))}")
    print(f"   First eigenvector std: {np.std(eigenvectors_dense[:, 0]):.10f}")
    
    print("\n2. Using scipy.sparse.linalg.eigsh (ARPACK):")
    # ARPACK for smallest eigenvalues
    k = 3  # Get first 3 eigenpairs
    eigenvalues_sparse, eigenvectors_sparse = eigsh(L_sparse, k=k, which='SM', tol=1e-6)
    # Sort by eigenvalue
    idx = eigenvalues_sparse.argsort()
    eigenvalues_sparse = eigenvalues_sparse[idx]
    eigenvectors_sparse = eigenvectors_sparse[:, idx]
    
    print(f"   First {k} eigenvalues: {eigenvalues_sparse}")
    print(f"   First eigenvector unique values: {len(np.unique(np.round(eigenvectors_sparse[:, 0], 6)))}")
    print(f"   First eigenvector std: {np.std(eigenvectors_sparse[:, 0]):.10f}")
    
    # Check if ARPACK gives constant eigenvector
    first_eigvec_sparse = eigenvectors_sparse[:, 0]
    is_constant = np.allclose(first_eigvec_sparse, first_eigvec_sparse[0], atol=1e-6)
    print(f"   First eigenvector is constant: {is_constant}")
    if is_constant:
        print(f"   Constant value: {first_eigvec_sparse[0]:.8f}")
        print(f"   Expected (1/sqrt(n)): {1/np.sqrt(len(first_eigvec_sparse)):.8f}")
    
    # Try with shift-invert mode (more accurate for small eigenvalues)
    print("\n3. Using ARPACK with shift-invert mode:")
    eigenvalues_si, eigenvectors_si = eigsh(L_sparse, k=k, sigma=0.0, which='LM', tol=1e-6)
    idx = eigenvalues_si.argsort()
    eigenvalues_si = eigenvalues_si[idx]
    eigenvectors_si = eigenvectors_si[:, idx]
    
    print(f"   First {k} eigenvalues: {eigenvalues_si}")
    print(f"   First eigenvector unique values: {len(np.unique(np.round(eigenvectors_si[:, 0], 6)))}")
    print(f"   First eigenvector std: {np.std(eigenvectors_si[:, 0]):.10f}")
    
    first_eigvec_si = eigenvectors_si[:, 0]
    is_constant_si = np.allclose(first_eigvec_si, first_eigvec_si[0], atol=1e-6)
    print(f"   First eigenvector is constant: {is_constant_si}")
    
    # Check connectivity
    print("\n4. Graph connectivity check:")
    print(f"   Min degree: {diag.min():.2f}")
    print(f"   Max degree: {diag.max():.2f}")
    print(f"   Zero degrees: {np.sum(diag < 1e-10)}")
    
    # The normalized Laplacian for a connected graph should have
    # smallest eigenvalue = 0 with constant eigenvector 1/sqrt(n)
    n = L_dense.shape[0]
    expected_eigvec = np.ones(n) / np.sqrt(n)
    
    print("\n5. Checking expected constant eigenvector:")
    L_times_const = L_dense @ expected_eigvec
    residual_norm = np.linalg.norm(L_times_const)
    print(f"   ||L * (1/sqrt(n))|| = {residual_norm:.10f}")
    print(f"   Is (1/sqrt(n)) an eigenvector with eigenvalue 0? {residual_norm < 1e-10}")

if __name__ == "__main__":
    compare_eigensolvers()