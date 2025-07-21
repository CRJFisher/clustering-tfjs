# Debugging Guide for Clustering Algorithms

This guide provides procedures and best practices for debugging clustering implementations, particularly when comparing against reference implementations like scikit-learn.

## Common Issues and Solutions

### 1. Test Results Don't Match Manual Runs

**Symptom**: Tests pass but manual execution of the same code fails.

**Possible Causes**:
- Compiled files in unexpected locations (e.g., `src/*.js`)
- Different import paths between tests and manual runs
- Stale build artifacts

**Solution**:
1. Check for compiled JS files in src directory: `find src -name "*.js"`
2. Ensure tests import from consistent location
3. Run `npm run build` before testing
4. Check `.gitignore` includes all compiled file patterns

### 2. Numerical Differences with Reference Implementation

**Symptom**: Results are close but not exactly matching sklearn.

**Possible Causes**:
- Different numerical precision (float32 vs float64)
- Different algorithm parameters or defaults
- Different random initialization

**Solution**:
1. Use debug mode to capture intermediate results:
   ```typescript
   const spectral = new SpectralClusteringModular({
     nClusters: 2,
     affinity: 'rbf',
     captureDebugInfo: true
   });
   ```
2. Compare step-by-step with reference:
   ```bash
   python tools/sklearn_comparison/compare_step_by_step.py
   ```

### 3. Disconnected Graph Components

**Symptom**: k-NN graphs create more connected components than desired clusters.

**Debugging Steps**:
1. Check number of components:
   ```python
   from scipy.sparse.csgraph import connected_components
   n_components, labels = connected_components(affinity_matrix)
   print(f"Number of components: {n_components}")
   ```
2. Visualize the data and components:
   ```bash
   python tools/debug/visualize_clustering.py
   ```

## Step-by-Step Debugging Procedures

### 1. Comparing Affinity Matrices

```typescript
// JavaScript/TypeScript
const spectral = new SpectralClusteringModular({ 
  nClusters: 2, 
  affinity: 'rbf',
  captureDebugInfo: true 
});
const affinity = await spectral.computeAffinityMatrix(X);
console.log('Affinity stats:', spectral.getDebugInfo()?.affinityStats);
```

```python
# Python (sklearn)
from sklearn.metrics.pairwise import rbf_kernel
affinity_sklearn = rbf_kernel(X, gamma=1.0)
print(f"Shape: {affinity_sklearn.shape}")
print(f"Non-zeros: {np.count_nonzero(affinity_sklearn)}")
print(f"Range: [{affinity_sklearn.min():.6f}, {affinity_sklearn.max():.6f}]")
```

### 2. Comparing Laplacians

```typescript
// Check Laplacian properties
const { laplacian } = await spectral.computeLaplacian(affinity);
const debugInfo = spectral.getDebugInfo();
console.log('First 5 eigenvalues:', debugInfo?.laplacianSpectrum?.slice(0, 5));
```

### 3. Comparing Embeddings

```typescript
// Compare embedding properties
const { embedding, eigenvalues } = await spectral.computeSpectralEmbedding(laplacian);
console.log('Embedding shape:', embedding.shape);
console.log('Unique values per dimension:', debugInfo?.embeddingStats?.uniqueValuesPerDim);
```

## Using Debug Scripts

### Compare Implementations

```bash
# Activate sklearn environment
source tools/sklearn_fixtures/.venv/bin/activate

# Run comprehensive comparison
python tools/debug/compare_implementations.py

# Output shows side-by-side comparison of:
# - Affinity matrix statistics
# - Laplacian eigenvalues
# - Embedding properties
# - Final clustering results
```

### Visualize Results

```bash
# Visualize clustering results
python tools/debug/visualize_clustering.py

# Creates plots showing:
# - True labels
# - Our predictions
# - Sklearn predictions
# - Saves to PNG files
```

### Analyze Eigenvectors

```bash
# Deep dive into eigenvector properties
python tools/debug/analyze_eigenvectors.py

# Shows:
# - Number of unique values per eigenvector
# - Component indicator patterns
# - Scaling factor effects
```

## Capturing Sklearn Intermediates

To capture intermediate results from sklearn for comparison:

```python
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster._spectral import spectral_embedding

# Get affinity matrix
affinity = rbf_kernel(X, gamma=1.0)

# Get spectral embedding
embedding = spectral_embedding(
    affinity, 
    n_components=n_clusters,
    drop_first=False,
    random_state=42
)

# Save for comparison
np.save('sklearn_affinity.npy', affinity)
np.save('sklearn_embedding.npy', embedding)
```

## Best Practices

1. **Always use debug mode** when investigating issues
2. **Compare at each step** rather than just final results
3. **Visualize the data** to understand the problem structure
4. **Check matrix properties** (symmetry, diagonal values, eigenvalues)
5. **Use consistent random seeds** for reproducibility
6. **Document findings** in task notes or comments

## Adding New Debug Tools

When creating new debug scripts:

1. Place in appropriate directory (`tools/debug/` or `tools/sklearn_comparison/`)
2. Include clear docstring explaining purpose
3. Make runnable from project root
4. Use relative paths for file access
5. Add usage example to this guide

## Troubleshooting Build Issues

### TypeScript Compilation

```bash
# Clean build
rm -rf dist/
npm run build

# Check for compilation errors
npx tsc --noEmit
```

### Test Runner Issues

```bash
# Clear Jest cache
npm test -- --clearCache

# Run with verbose output
npm test -- --verbose

# Run specific test file
npm test -- test/integration/spectral_steps.test.ts
```

## Performance Profiling

To identify performance bottlenecks:

```typescript
console.time('affinity');
const affinity = await spectral.computeAffinityMatrix(X);
console.timeEnd('affinity');

console.time('laplacian');
const { laplacian } = await spectral.computeLaplacian(affinity);
console.timeEnd('laplacian');

console.time('embedding');
const { embedding } = await spectral.computeSpectralEmbedding(laplacian);
console.timeEnd('embedding');
```

## Memory Management

Monitor tensor memory usage:

```typescript
console.log('Memory before:', tf.memory());
// ... run algorithm ...
console.log('Memory after:', tf.memory());

// Ensure proper cleanup
spectral.dispose();
```