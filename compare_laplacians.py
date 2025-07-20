import numpy as np
import json
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse.csgraph import laplacian

# Load the same test data
with open('test/fixtures/spectral/circles_n2_rbf.json', 'r') as f:
    data = json.load(f)

X = np.array(data['X'])
params = data['params']

# Compute affinity matrix
gamma = params.get('gamma', 1.0)
affinity = rbf_kernel(X, gamma=gamma)

# Compute Laplacian
lap, dd = laplacian(affinity, normed=True, return_diag=True)

# Save for comparison
np.save('tmp/sklearn_laplacian.npy', lap)
np.save('tmp/sklearn_dd.npy', dd)

print("Laplacian shape:", lap.shape)
print("Laplacian diagonal (first 5):", np.diag(lap)[:5])
print("Laplacian first row (first 5 non-diag):", lap[0, 1:6])
print("\ndd (degree^0.5) first 5:", dd[:5])