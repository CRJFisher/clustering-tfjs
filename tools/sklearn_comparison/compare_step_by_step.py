#!/usr/bin/env python3
"""
Comprehensive comparison of sklearn SpectralClustering with our implementation.
This script tests with the exact parameters and data used in the fixture tests.
"""

import numpy as np
import json
import os
import sys
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
import warnings

# Path to sklearn venv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools/sklearn_fixtures/.venv/lib/python3.9/site-packages'))

# Silence warnings for cleaner output
warnings.filterwarnings('ignore')

def load_fixture(fixture_name):
    """Load a fixture file."""
    path = f"test/fixtures/spectral/{fixture_name}"
    with open(path, 'r') as f:
        return json.load(f)

def compute_affinity_matrix(X, affinity, gamma=None, n_neighbors=10):
    """Compute affinity matrix exactly as sklearn does."""
    if affinity == 'rbf':
        if gamma is None:
            gamma = 1.0 / X.shape[1]
        return rbf_kernel(X, gamma=gamma)
    elif affinity == 'nearest_neighbors':
        # sklearn uses mode='connectivity' by default
        return kneighbors_graph(X, n_neighbors=n_neighbors, mode='connectivity', include_self=True)
    else:
        raise ValueError(f"Unknown affinity: {affinity}")

def analyze_laplacian(affinity_matrix, n_clusters):
    """Analyze the Laplacian matrix and its eigendecomposition."""
    # Convert to dense if sparse
    if hasattr(affinity_matrix, 'toarray'):
        W = affinity_matrix.toarray()
    else:
        W = affinity_matrix
    
    # Compute normalized Laplacian as sklearn does
    # 1. Zero out diagonal
    np.fill_diagonal(W, 0)
    
    # 2. Compute degrees
    degrees = np.sum(W, axis=1)
    
    # 3. Handle zero degrees
    sqrt_degrees = np.sqrt(degrees)
    sqrt_degrees[degrees == 0] = 1
    
    # 4. Compute D^(-1/2)
    D_sqrt_inv = np.diag(1.0 / sqrt_degrees)
    
    # 5. Normalized Laplacian: L = I - D^(-1/2) * W * D^(-1/2)
    normalized_W = D_sqrt_inv @ W @ D_sqrt_inv
    L = np.eye(W.shape[0]) - normalized_W
    
    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    # Sort by eigenvalue
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return {
        'affinity_matrix': W,
        'degrees': degrees,
        'laplacian': L,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'sqrt_degrees': sqrt_degrees
    }

def analyze_spectral_embedding(eigenvectors, eigenvalues, sqrt_degrees, n_clusters):
    """Analyze the spectral embedding step."""
    # Select first n_clusters eigenvectors
    embedding = eigenvectors[:, :n_clusters].copy()
    
    # Apply diffusion map scaling: scale by sqrt(1 - eigenvalue)
    scaling_factors = np.sqrt(np.maximum(0, 1 - eigenvalues[:n_clusters]))
    embedding_scaled = embedding * scaling_factors
    
    # Check if eigenvectors are constant
    is_constant = []
    for i in range(n_clusters):
        unique_vals = np.unique(np.round(embedding[:, i], 10))
        is_constant.append(len(unique_vals) == 1)
    
    return {
        'embedding_raw': embedding,
        'embedding_scaled': embedding_scaled,
        'scaling_factors': scaling_factors,
        'is_constant': is_constant,
        'unique_values_per_dim': [len(np.unique(np.round(embedding[:, i], 10))) for i in range(n_clusters)]
    }

def main():
    """Run comprehensive comparison for all fixtures."""
    fixture_files = [
        'blobs_n2_knn.json', 'blobs_n2_rbf.json', 
        'blobs_n3_knn.json', 'blobs_n3_rbf.json',
        'circles_n2_knn.json', 'circles_n2_rbf.json',
        'circles_n3_knn.json', 'circles_n3_rbf.json',
        'moons_n2_knn.json', 'moons_n2_rbf.json',
        'moons_n3_knn.json', 'moons_n3_rbf.json'
    ]
    
    results = {}
    
    for fixture_file in fixture_files:
        print(f"\n{'='*60}")
        print(f"Analyzing {fixture_file}")
        print('='*60)
        
        # Load fixture
        fixture = load_fixture(fixture_file)
        X = np.array(fixture['X'])
        y_true = np.array(fixture['labels'])
        params = fixture['params']
        
        # Extract parameters
        n_clusters = params['nClusters']
        affinity = params['affinity']
        gamma = params.get('gamma', None)
        n_neighbors = params.get('nNeighbors', 10)
        random_state = params.get('randomState', 42)
        
        print(f"\nParameters:")
        print(f"  n_clusters: {n_clusters}")
        print(f"  affinity: {affinity}")
        print(f"  gamma: {gamma}")
        print(f"  n_neighbors: {n_neighbors}")
        print(f"  random_state: {random_state}")
        print(f"  n_samples: {X.shape[0]}")
        
        # Run sklearn
        kwargs = {
            'n_clusters': n_clusters,
            'affinity': affinity,
            'random_state': random_state,
            'n_init': 10,
            'assign_labels': 'kmeans'
        }
        
        # Only add gamma for RBF affinity
        if affinity == 'rbf' and gamma is not None:
            kwargs['gamma'] = gamma
        
        # Only add n_neighbors for k-NN affinity
        if affinity == 'nearest_neighbors' and n_neighbors is not None:
            kwargs['n_neighbors'] = n_neighbors
            
        model = SpectralClustering(**kwargs)
        y_pred = model.fit_predict(X)
        
        # Compute ARI
        ari = adjusted_rand_score(y_true, y_pred)
        print(f"\nSklearn ARI: {ari:.6f}")
        
        # Detailed analysis
        affinity_matrix = compute_affinity_matrix(X, affinity, gamma, n_neighbors)
        laplacian_info = analyze_laplacian(affinity_matrix, n_clusters)
        
        # Count components
        num_zero_eigenvals = np.sum(laplacian_info['eigenvalues'] < 1e-10)
        print(f"\nGraph structure:")
        print(f"  Number of connected components: {num_zero_eigenvals}")
        print(f"  First {n_clusters+2} eigenvalues: {laplacian_info['eigenvalues'][:n_clusters+2]}")
        
        # Analyze embedding
        embedding_info = analyze_spectral_embedding(
            laplacian_info['eigenvectors'],
            laplacian_info['eigenvalues'],
            laplacian_info['sqrt_degrees'],
            n_clusters
        )
        
        print(f"\nEmbedding analysis:")
        print(f"  Scaling factors: {embedding_info['scaling_factors']}")
        print(f"  Constant eigenvectors: {embedding_info['is_constant']}")
        print(f"  Unique values per dimension: {embedding_info['unique_values_per_dim']}")
        
        # Store results
        results[fixture_file] = {
            'ari': ari,
            'n_components': num_zero_eigenvals,
            'eigenvalues': laplacian_info['eigenvalues'][:n_clusters+2].tolist(),
            'scaling_factors': embedding_info['scaling_factors'].tolist(),
            'is_constant': embedding_info['is_constant'],
            'params': params
        }
        
        # Special analysis for failing k-NN cases
        if affinity == 'nearest_neighbors' and ari < 0.95:
            print(f"\n[SPECIAL ANALYSIS - Failing k-NN case]")
            print(f"  Examining why sklearn achieves ARI={ari:.3f}")
            
            # Check affinity matrix structure
            if hasattr(affinity_matrix, 'toarray'):
                W = affinity_matrix.toarray()
            else:
                W = affinity_matrix
            
            # Check connectivity
            print(f"  Affinity matrix density: {np.count_nonzero(W) / W.size:.3f}")
            print(f"  Min degree: {laplacian_info['degrees'].min()}")
            print(f"  Max degree: {laplacian_info['degrees'].max()}")
            
            # Save embedding for detailed comparison
            np.save(f"debug_{fixture_file.replace('.json', '_embedding.npy')}", 
                    embedding_info['embedding_scaled'])
    
    # Summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    
    passing = []
    failing = []
    
    for fixture, result in results.items():
        if result['ari'] >= 0.95:
            passing.append(fixture)
        else:
            failing.append(fixture)
    
    print(f"\nPassing tests ({len(passing)}/12):")
    for f in passing:
        print(f"  ✓ {f} (ARI={results[f]['ari']:.3f})")
    
    print(f"\nFailing tests ({len(failing)}/12):")
    for f in failing:
        print(f"  ✗ {f} (ARI={results[f]['ari']:.3f}, components={results[f]['n_components']})")
    
    # Convert numpy types to native Python types for JSON serialization
    results_json = {}
    for k, v in results.items():
        results_json[k] = {
            'ari': float(v['ari']),
            'n_components': int(v['n_components']),
            'eigenvalues': [float(x) for x in v['eigenvalues']],
            'scaling_factors': [float(x) for x in v['scaling_factors']],
            'is_constant': v['is_constant'],
            'params': v['params']
        }
    
    # Save detailed results
    with open('sklearn_exact_comparison.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\nDetailed results saved to sklearn_exact_comparison.json")

if __name__ == "__main__":
    main()