import numpy as np
import json
from sklearn.cluster import SpectralClustering
from scipy.sparse.csgraph import laplacian

def compare_affinity_matrices():
    print("=== Comparing affinity matrices ===\n")
    
    # Load moons dataset
    with open('test/fixtures/spectral/moons_n2_knn.json', 'r') as f:
        fixture = json.load(f)
    
    X = np.array(fixture['X'])
    params = fixture['params']
    
    # Create affinity matrix with sklearn
    sc = SpectralClustering(
        n_clusters=params['nClusters'],
        affinity='nearest_neighbors',
        n_neighbors=params['nNeighbors'],
        random_state=params['randomState']
    )
    sc.fit(X)
    sklearn_affinity = sc.affinity_matrix_
    
    print("sklearn affinity matrix:")
    print(f"  Shape: {sklearn_affinity.shape}")
    print(f"  Type: {type(sklearn_affinity)}")
    print(f"  Diagonal sum: {sklearn_affinity.diagonal().sum():.6f}")
    
    # Check symmetry
    if hasattr(sklearn_affinity, 'todense'):
        A_dense = sklearn_affinity.todense()
    else:
        A_dense = sklearn_affinity
    
    is_symmetric = np.allclose(A_dense, A_dense.T)
    print(f"  Is symmetric: {is_symmetric}")
    
    # Check row sums (degrees)
    degrees = np.array(A_dense.sum(axis=1)).flatten()
    print(f"  Degree range: [{degrees.min():.2f}, {degrees.max():.2f}]")
    print(f"  Mean degree: {degrees.mean():.2f}")
    
    # Check normalized Laplacian
    L, diag = laplacian(sklearn_affinity, normed=True, return_diag=True)
    if hasattr(L, 'todense'):
        L_dense = L.todense()
    else:
        L_dense = L
    
    print("\nNormalized Laplacian check:")
    
    # For a connected graph, constant vector should be eigenvector
    n = L_dense.shape[0]
    ones = np.ones(n)
    
    # L * 1 = ?
    L_times_ones = L_dense @ ones
    print(f"  ||L * 1|| = {np.linalg.norm(L_times_ones):.10f}")
    
    # For normalized Laplacian, we need L * (D^(1/2) * 1)
    sqrt_degrees = np.sqrt(degrees)
    scaled_ones = sqrt_degrees
    L_times_scaled = L_dense @ scaled_ones
    print(f"  ||L * D^(1/2)1|| = {np.linalg.norm(L_times_scaled):.10f}")
    
    # The actual eigenvector should be 1/sqrt(n) for all entries
    const_eigvec = np.ones(n) / np.sqrt(n)
    L_times_const = L_dense @ const_eigvec
    print(f"  ||L * (1/sqrt(n))|| = {np.linalg.norm(L_times_const):.10f}")
    
    # Check specific entries of affinity matrix
    print("\nSample affinity values:")
    print("  First row non-zero entries:")
    row0 = np.array(A_dense[0, :]).flatten()
    nonzero_idx = np.where(row0 > 0)[0]
    for idx in nonzero_idx[:5]:
        print(f"    A[0, {idx}] = {row0[idx]:.6f}")
    
    # Check if it's binary (0/1) or weighted
    unique_vals = np.unique(A_dense[A_dense > 0])
    print(f"\n  Unique non-zero values: {len(unique_vals)}")
    if len(unique_vals) < 10:
        print(f"    Values: {unique_vals}")
    
    # Save affinity matrix for comparison
    np.save('sklearn_affinity_moons.npy', A_dense)
    print("\nSaved sklearn affinity matrix to sklearn_affinity_moons.npy")

if __name__ == "__main__":
    compare_affinity_matrices()