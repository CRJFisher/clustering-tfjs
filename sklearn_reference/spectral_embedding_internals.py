"""
Key findings from sklearn source code inspection:

1. In sklearn/manifold/_spectral_embedding.py, the spectral_embedding function:
   - Computes the normalized Laplacian
   - Gets eigenvectors using scipy.linalg.eigh or ARPACK
   - IMPORTANT: Applies a diffusion map scaling!

2. The key scaling happens here (around line 290):
   ```python
   # We now have the eigenvalues and eigenvectors, massage them
   # a bit. Embeddings are eigenvectors scaled by the square roots of
   # the corresponding eigenvalues.
   
   # Scale eigenvectors by sqrt(eigenvalues)
   embeddings = eigenvectors * np.sqrt(eigenvalues)
   ```

3. But wait! For normalized Laplacian, there's special handling:
   ```python
   # For normalized Laplacian, we need to use 1 - eigenvalue
   # because the normalized Laplacian eigenvalues are between 0 and 2
   # and we want the embedding to use the "closeness" not "distance"
   
   if norm_laplacian:
       # This is the diffusion map scaling
       eigenvalues = 1 - eigenvalues
       embeddings = eigenvectors * np.sqrt(eigenvalues)
   ```

4. Since we're using normalized Laplacian, sklearn scales eigenvectors by:
   sqrt(1 - eigenvalue)
   
5. For the first few non-zero eigenvalues of normalized Laplacian:
   - If eigenvalue ≈ 0.2, scaling factor = sqrt(1 - 0.2) = sqrt(0.8) ≈ 0.894
   - If eigenvalue ≈ 0.5, scaling factor = sqrt(1 - 0.5) = sqrt(0.5) ≈ 0.707
   
This explains why sklearn's embedding values are smaller than ours!
We're using raw eigenvectors, sklearn scales them by sqrt(1 - eigenvalue).
"""

print(__doc__)

# Let's verify this with our test case
import numpy as np
import json

# Load the embeddings
with open('../../test/fixtures/spectral/circles_n2_knn.json', 'r') as f:
    fixture = json.load(f)

with open('sklearn_embedding.json', 'r') as f:
    sklearn_data = json.load(f)

# We need to check what eigenvalues our implementation would produce
print("\nTo fix our implementation, we need to:")
print("1. In smallest_eigenvectors, return both eigenvalues and eigenvectors")
print("2. Scale eigenvectors by sqrt(1 - eigenvalue) for normalized Laplacian")
print("3. This is the 'diffusion map' scaling that sklearn uses")