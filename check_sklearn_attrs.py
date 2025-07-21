from sklearn.cluster import SpectralClustering
import numpy as np

# Create a simple example
X = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])
sc = SpectralClustering(n_clusters=2)
sc.fit(X)

print("SpectralClustering attributes after fit:")
for attr in dir(sc):
    if not attr.startswith('_') and hasattr(sc, attr):
        val = getattr(sc, attr)
        if not callable(val):
            print(f"  {attr}: {type(val)}")
            if hasattr(val, 'shape'):
                print(f"    shape: {val.shape}")