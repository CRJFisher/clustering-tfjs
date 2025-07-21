import numpy as np
import json
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
from sklearn.manifold import spectral_embedding
import subprocess

# Test on one failing case
fixture_name = 'circles_n2_rbf'

print(f"Detailed eigenvector comparison for {fixture_name}:\n")

with open(f'./test/fixtures/spectral/{fixture_name}.json') as f:
    data = json.load(f)

X = np.array(data['X'])
n_samples = X.shape[0]

# Create affinity matrix
if 'rbf' in fixture_name:
    affinity = rbf_kernel(X, gamma=1.0)
else:
    affinity = kneighbors_graph(X, n_neighbors=10, mode='connectivity', include_self=True)
    affinity = 0.5 * (affinity + affinity.T)

# Get normalized Laplacian
L = laplacian(affinity, normed=True)

print("1. Matrix properties:")
print(f"   Affinity shape: {affinity.shape}")
print(f"   Laplacian type: normalized")

# sklearn's approach
print("\n2. sklearn's eigendecomposition:")
eigenvalues_sk, eigenvectors_sk = eigh(L)
print(f"   First 5 eigenvalues: {eigenvalues_sk[:5]}")
print(f"   Eigenvalue range: [{eigenvalues_sk.min():.6e}, {eigenvalues_sk.max():.6f}]")

# Check eigenvector properties
print("\n3. sklearn eigenvector analysis:")
for i in range(min(3, n_samples)):
    vec = eigenvectors_sk[:, i]
    print(f"   Eigenvector {i}:")
    print(f"     - L2 norm: {np.linalg.norm(vec):.6f}")
    print(f"     - Range: [{vec.min():.6f}, {vec.max():.6f}]")
    print(f"     - Std dev: {vec.std():.6f}")
    print(f"     - First 5 values: {vec[:5]}")

# Get sklearn's spectral embedding
embedding_sk = spectral_embedding(affinity, n_components=2, drop_first=False, random_state=42)
print("\n4. sklearn spectral embedding (with diffusion scaling):")
for i in range(2):
    vec = embedding_sk[:, i]
    print(f"   Dimension {i}:")
    print(f"     - L2 norm: {np.linalg.norm(vec):.6f}")
    print(f"     - Range: [{vec.min():.6f}, {vec.max():.6f}]")
    print(f"     - Std dev: {vec.std():.6f}")

# Now let's see what our implementation produces
print("\n5. Running our implementation to extract eigenvectors...")

# Create a test script to extract our eigenvectors
test_script = '''
import * as tf from "@tensorflow/tfjs-node";
import { improved_jacobi_eigen } from "./src/utils/eigen_improved";
import { normalised_laplacian } from "./src/utils/laplacian";
import { compute_rbf_affinity } from "./src/utils/affinity";
import { smallest_eigenvectors_with_values } from "./src/utils/smallest_eigenvectors_with_values";
import * as fs from "fs";

async function main() {
  const fixture = JSON.parse(fs.readFileSync("./test/fixtures/spectral/circles_n2_rbf.json", "utf8"));
  const X = tf.tensor2d(fixture.X);
  
  // Create affinity
  const affinity = compute_rbf_affinity(X, 1.0);
  
  // Get Laplacian
  const L = normalised_laplacian(affinity);
  
  // Get our eigenvectors
  const { eigenvectors, eigenvalues } = smallest_eigenvectors_with_values(L, 2);
  
  // Also run full decomposition for comparison
  const Larray = await L.array();
  const { eigenvalues: allEvals, eigenvectors: allEvecs } = improved_jacobi_eigen(
    tf.tensor2d(Larray),
    { isPSD: true, maxIterations: 3000, tolerance: 1e-14 }
  );
  
  // Save results
  const results = {
    selectedEigenvalues: await eigenvalues.array(),
    selectedEigenvectors: await eigenvectors.array(),
    allEigenvalues: allEvals,
    firstThreeEigenvectors: allEvecs.map(row => row.slice(0, 3))
  };
  
  fs.writeFileSync("our_eigen_results.json", JSON.stringify(results, null, 2));
  console.log("Saved eigenvector results to our_eigen_results.json");
  
  // Cleanup
  X.dispose();
  affinity.dispose();
  L.dispose();
  eigenvectors.dispose();
  eigenvalues.dispose();
}

main().catch(console.error);
'''

with open('test_our_eigen.ts', 'w') as f:
    f.write(test_script)

# Run it
result = subprocess.run(['npx', 'ts-node', 'test_our_eigen.ts'], capture_output=True, text=True)
if result.returncode != 0:
    print(f"Error running our implementation: {result.stderr}")
else:
    print(result.stdout)

# Load and analyze our results
try:
    with open('our_eigen_results.json') as f:
        our_results = json.load(f)
    
    print("\n6. Our eigendecomposition results:")
    print(f"   Selected eigenvalues: {our_results['selectedEigenvalues']}")
    print(f"   All eigenvalues (first 5): {our_results['allEigenvalues'][:5]}")
    
    print("\n7. Comparing eigenvectors:")
    our_evecs = np.array(our_results['firstThreeEigenvectors'])
    
    for i in range(min(3, our_evecs.shape[1])):
        our_vec = our_evecs[:, i]
        sk_vec = eigenvectors_sk[:, i]
        
        # Check alignment (might be flipped)
        dot_prod = np.abs(np.dot(our_vec, sk_vec))
        
        print(f"\n   Eigenvector {i}:")
        print(f"     Our norm: {np.linalg.norm(our_vec):.6f}, sklearn norm: {np.linalg.norm(sk_vec):.6f}")
        print(f"     Alignment (abs dot product): {dot_prod:.6f}")
        print(f"     Our range: [{our_vec.min():.6f}, {our_vec.max():.6f}]")
        print(f"     sklearn range: [{sk_vec.min():.6f}, {sk_vec.max():.6f}]")
        print(f"     Difference in std dev: {abs(our_vec.std() - sk_vec.std()):.6f}")
        
except Exception as e:
    print(f"Could not load our results: {e}")

# Key metrics to check
print("\n8. Key differences to investigate:")
print("   - Eigenvalue accuracy (especially near zero)")
print("   - Eigenvector normalization")
print("   - Zero eigenvalue threshold (we use 1e-2, sklearn might use smaller)")
print("   - Numerical precision throughout the pipeline")