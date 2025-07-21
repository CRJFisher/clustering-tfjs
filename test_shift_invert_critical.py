import numpy as np
import json
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
import scipy.sparse as sp

# Load data
with open('./test/fixtures/spectral/blobs_n2_knn.json', 'r') as f:
    data = json.load(f)

X = np.array(data['X'])

# Create affinity matrix
affinity = kneighbors_graph(X, n_neighbors=10, mode='connectivity', include_self=True)
affinity = 0.5 * (affinity + affinity.T)

# Get normalized Laplacian
L, dd = laplacian(affinity, normed=True, return_diag=True)

print("Testing different eigenvalue approaches:\n")

# 1. Standard eigendecomposition
eigenvalues_std, eigenvectors_std = np.linalg.eigh(L.toarray())
print("1. Standard eigendecomposition:")
print(f"   First 3 eigenvalues: {eigenvalues_std[:3]}")
for i in range(3):
    vec = eigenvectors_std[:, i]
    unique = len(np.unique(np.round(vec, 6)))
    print(f"   Eigenvector {i}: {unique} unique values")

# 2. ARPACK without shift-invert (what we might be doing)
try:
    L_csr = L.tocsr()
    eigenvalues_arpack, eigenvectors_arpack = eigsh(L_csr, k=3, which='SM')
    print("\n2. ARPACK with which='SM' (smallest magnitude):")
    print(f"   Eigenvalues: {eigenvalues_arpack}")
    for i in range(3):
        vec = eigenvectors_arpack[:, i]
        unique = len(np.unique(np.round(vec, 6)))
        print(f"   Eigenvector {i}: {unique} unique values")
except Exception as e:
    print(f"\n2. ARPACK failed: {e}")

# 3. Shift-invert mode (what sklearn does)
try:
    # sklearn computes eigenvalues of -L with sigma=1.0
    L_neg = -L_csr
    eigenvalues_si, eigenvectors_si = eigsh(L_neg, k=6, sigma=1.0, which='LM')
    print("\n3. Shift-invert mode (sklearn's approach):")
    print(f"   Eigenvalues (of -L): {eigenvalues_si}")
    # Reverse order to match sklearn
    for i in range(3):
        idx = 5 - i
        vec = eigenvectors_si[:, idx]
        unique = len(np.unique(np.round(vec, 6)))
        print(f"   Eigenvector {i}: {unique} unique values")
        if unique <= 5:
            print(f"      First 10 values: {vec[:10]}")
except Exception as e:
    print(f"\n3. Shift-invert failed: {e}")

# The key question: Can we get component indicators WITHOUT shift-invert?
print("\n\nAlternative approaches:")

# Maybe we need to look at the nullspace differently?
from scipy.linalg import null_space
try:
    # For disconnected graphs, L has a nullspace dimension = number of components
    nullspace = null_space(L.toarray())
    print(f"\n4. Nullspace approach:")
    print(f"   Nullspace dimension: {nullspace.shape[1]}")
    for i in range(min(3, nullspace.shape[1])):
        vec = nullspace[:, i]
        unique = len(np.unique(np.round(vec, 6)))
        print(f"   Nullspace vector {i}: {unique} unique values")
except Exception as e:
    print(f"\n4. Nullspace failed: {e}")

# What about using the graph structure directly?
from scipy.sparse.csgraph import connected_components
n_comp, labels = connected_components(affinity)
print(f"\n5. Direct component detection:")
print(f"   Number of components: {n_comp}")
print(f"   Component labels: {np.unique(labels)}")
print(f"   This is what our component indicator approach uses!")

# Final check: Is there ANY way to get component indicators from standard eigen?
print("\n\nFinal analysis:")
print("Standard eigenvectors have many unique values because they encode")
print("spectral information beyond just component membership.")
print("Shift-invert specifically targets near-zero eigenvalues, producing")
print("clean component indicators.")
print("\nConclusion: Component indicators OR shift-invert are needed.")