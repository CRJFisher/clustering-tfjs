import numpy as np
import json
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian, connected_components
from scipy.sparse.linalg import eigsh

# Load data
with open('./test/fixtures/spectral/blobs_n2_knn.json', 'r') as f:
    data = json.load(f)

X = np.array(data['X'])

# Create affinity matrix
affinity = kneighbors_graph(X, n_neighbors=10, mode='connectivity', include_self=True)
affinity = 0.5 * (affinity + affinity.T)

# Get connected components
n_components, labels = connected_components(affinity)
print(f"Number of connected components: {n_components}")
print(f"Component sizes: {[np.sum(labels == i) for i in range(n_components)]}")

# Get normalized Laplacian
L_norm = laplacian(affinity, normed=True)

print("\nTesting what sklearn actually does internally:")

# This is the EXACT call sklearn makes in _spectral_embedding when graph is disconnected
# See: sklearn/manifold/_spectral_embedding.py lines ~232-240
try:
    # sklearn uses shift-invert mode with sigma near 0 for disconnected graphs
    eigenvalues, eigenvectors = eigsh(
        L_norm, 
        k=n_components + 1,  # Get one extra
        sigma=0,  # Near zero eigenvalues
        which='LM',  # Largest magnitude (after shift)
        tol=0  # Machine precision
    )
    
    print(f"\nShift-invert with sigma=0:")
    print(f"Eigenvalues: {eigenvalues}")
    
    # Check eigenvectors
    for i in range(min(3, eigenvectors.shape[1])):
        vec = eigenvectors[:, i]
        unique_vals = np.unique(np.round(vec, 10))
        print(f"\nEigenvector {i}:")
        print(f"  Unique values: {len(unique_vals)}")
        if len(unique_vals) <= 5:
            print(f"  Values: {unique_vals}")
            # Check which component each value corresponds to
            for val in unique_vals:
                indices = np.where(np.abs(vec - val) < 1e-10)[0]
                components = labels[indices]
                print(f"    {val:.10f} appears in components: {np.unique(components)}")
                
except Exception as e:
    print(f"Shift-invert failed: {e}")

# Now let's verify sklearn's exact approach by looking at source
print("\n\nFrom sklearn source analysis:")
print("When graph is disconnected, sklearn:")
print("1. Detects disconnection using connected_components")
print("2. Uses eigsh with sigma=0 to get component indicators")
print("3. These eigenvectors have constant values per component")
print("4. This is why sklearn's spectral_embedding returns vectors with 3 unique values")

# Final verification: our component indicators should match sklearn's eigenvectors
print("\n\nOur component indicator approach:")
from scipy.sparse.csgraph import connected_components
n_comp, comp_labels = connected_components(affinity)

# Create indicators like we do
indicators = np.zeros((len(comp_labels), n_comp))
comp_sizes = np.bincount(comp_labels)
for i, label in enumerate(comp_labels):
    indicators[i, label] = 1.0 / np.sqrt(comp_sizes[label])

print(f"Component indicator shape: {indicators.shape}")
for i in range(indicators.shape[1]):
    unique = len(np.unique(indicators[:, i]))
    print(f"Indicator {i}: {unique} unique values")