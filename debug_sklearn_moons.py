import numpy as np
import json
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
from scipy.sparse.csgraph import laplacian
from sklearn.manifold import spectral_embedding
from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances

def load_fixture(filename):
    with open(f'test/fixtures/spectral/{filename}', 'r') as f:
        fixture = json.load(f)
    return fixture

def analyze_moons_clustering():
    print("=== Analyzing sklearn's behavior on moons datasets ===\n")
    
    datasets = ['moons_n2_knn.json', 'moons_n2_rbf.json']
    
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
        eigenvalues = np.sort(np.linalg.eigvalsh(L_dense))[:10]
        print(f"First 10 eigenvalues: {eigenvalues}")
        
        # Check if graph is disconnected
        n_components = np.sum(eigenvalues < 1e-8)
        print(f"Number of zero eigenvalues (components): {n_components}")
        
        # Get spectral embedding manually
        embedding = spectral_embedding(
            affinity_matrix,
            n_components=params['nClusters'],
            drop_first=False,  # SpectralClustering uses False
            eigen_solver='arpack',
            random_state=params['randomState']
        )
        
        print(f"Embedding shape: {embedding.shape}")
        
        # Check if eigenvector recovery is applied
        # Get eigenvectors of normalized Laplacian
        eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
        idx = eigenvalues.argsort()
        eigenvectors = eigenvectors[:, idx[:params['nClusters']]]
        
        # Check unique values before recovery
        print("\nBefore recovery:")
        for i in range(min(3, eigenvectors.shape[1])):
            unique_vals = len(np.unique(np.round(eigenvectors[:, i], 6)))
            print(f"  Eigenvector {i}: {unique_vals} unique values")
        
        # Apply recovery
        sqrt_diag = np.sqrt(diag)
        eigenvectors_recovered = eigenvectors / sqrt_diag[:, np.newaxis]
        
        print("\nAfter recovery:")
        for i in range(min(3, eigenvectors_recovered.shape[1])):
            unique_vals = len(np.unique(np.round(eigenvectors_recovered[:, i], 6)))
            print(f"  Eigenvector {i}: {unique_vals} unique values")
        
        # Compare with sklearn's embedding
        print("\nsklearn's embedding unique values:")
        for i in range(min(3, embedding.shape[1])):
            unique_vals = len(np.unique(np.round(embedding[:, i], 6)))
            print(f"  Embedding dim {i}: {unique_vals} unique values")
        
        # Check if sklearn's embedding matches recovered eigenvectors
        # Note: may need to account for sign differences
        print("\nCorrelations:")
        for i in range(min(embedding.shape[1], eigenvectors_recovered.shape[1])):
            # Flatten arrays to ensure 1D
            emb_col = embedding[:, i].flatten()
            eig_col = eigenvectors_recovered[:, i].flatten()
            
            # Check if constant (avoid division by zero in correlation)
            if np.std(emb_col) < 1e-10 or np.std(eig_col) < 1e-10:
                print(f"  Embedding[{i}] or eigenvector[{i}] is constant")
            else:
                correlation = np.abs(np.corrcoef(emb_col, eig_col)[0, 1])
                print(f"  Correlation between sklearn embedding[{i}] and recovered eigenvector[{i}]: {correlation:.4f}")

if __name__ == "__main__":
    analyze_moons_clustering()