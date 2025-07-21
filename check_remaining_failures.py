import json
from pathlib import Path
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.sparse.csgraph import connected_components
import numpy as np

# List all fixture files
fixture_dir = Path('./test/fixtures/spectral')
fixtures = sorted(fixture_dir.glob('*.json'))

print("Checking connectivity of all fixtures:\n")
print(f"{'Fixture':<20} {'Connected?':<12} {'Components':<12} {'Status'}")
print("-" * 60)

failing_connected = []
failing_disconnected = []

for fixture_path in fixtures:
    with open(fixture_path) as f:
        data = json.load(f)
    
    name = fixture_path.stem
    X = np.array(data['X'])
    y_true = np.array(data['labels'])  # sklearn fixture format
    
    # Check connectivity based on affinity type
    if 'knn' in name:
        affinity = kneighbors_graph(X, n_neighbors=10, mode='connectivity', include_self=True)
        affinity = 0.5 * (affinity + affinity.T)
    else:  # rbf
        from sklearn.metrics.pairwise import rbf_kernel
        affinity = rbf_kernel(X, gamma=1.0)
        # Convert to sparse and threshold
        from scipy import sparse
        affinity[affinity < 1e-10] = 0
        affinity = sparse.csr_matrix(affinity)
    
    n_comp, _ = connected_components(affinity)
    is_connected = n_comp == 1
    
    # Get expected ARI from our test results
    # Based on the summary, these are passing:
    passing = ['blobs_n3_knn', 'blobs_n3_rbf', 'circles_n2_knn', 'moons_n2_knn', 
               'moons_n3_knn', 'blobs_n2_knn', 'blobs_n2_rbf']
    
    status = "PASS" if name in passing else "FAIL"
    
    print(f"{name:<20} {str(is_connected):<12} {n_comp:<12} {status}")
    
    if status == "FAIL":
        if is_connected:
            failing_connected.append(name)
        else:
            failing_disconnected.append(name)

print(f"\nSummary:")
print(f"Passing tests: {len(passing)}/12")
print(f"Failing connected graphs: {len(failing_connected)} - {failing_connected}")
print(f"Failing disconnected graphs: {len(failing_disconnected)} - {failing_disconnected}")

print("\nAnalysis of remaining failures:")
for name in failing_connected + failing_disconnected:
    print(f"\n{name}:")
    with open(f'./test/fixtures/spectral/{name}.json') as f:
        data = json.load(f)
    X = np.array(data['X'])
    print(f"  Shape: {X.shape}")
    print(f"  n_clusters: {len(np.unique(data['labels']))}")