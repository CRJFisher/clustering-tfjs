# clustering-tfjs

[![npm version](https://badge.fury.io/js/clustering-tfjs.svg)](https://www.npmjs.com/package/clustering-tfjs)
[![Build Status](https://github.com/CRJFisher/clustering-js/workflows/CI/badge.svg)](https://github.com/CRJFisher/clustering-js/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/TypeScript-Ready-blue.svg)](https://www.typescriptlang.org/)

Native TypeScript implementation of clustering algorithms powered by TensorFlow.js with full browser and Node.js support.

## Features

- ✅ Pure TypeScript/JavaScript (no Python required)
- ✅ Multiple clustering algorithms (K-Means, Spectral, Agglomerative, SOM)
- ✅ Powered by TensorFlow.js for performance
- ✅ **Works in both Node.js and browsers**
- ✅ Platform-optimized bundles (49KB for browser, 163KB for Node.js)
- ✅ TypeScript support with full type definitions
- ✅ GPU acceleration available (WebGL in browser, CUDA in Node.js)
- ✅ Automatic backend selection
- ✅ Extensively tested for parity with scikit-learn

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Algorithms](#algorithms)
4. [Validation Metrics](#validation-metrics)
5. [Backend Selection](#backend-selection)
6. [API Reference](#api-reference)
7. [Examples](#examples)
8. [Performance](#performance)
9. [Migration from scikit-learn](#migration-from-scikit-learn)
10. [Contributing](#contributing)
11. [License](#license)

## Quick Start

### Install

```bash
# For Node.js with acceleration
npm install clustering-tfjs @tensorflow/tfjs-node

# For Node.js with GPU support
npm install clustering-tfjs @tensorflow/tfjs-node-gpu

# For browser usage (TensorFlow.js loaded separately)
npm install clustering-tfjs
```

> **Note**: For Windows users or if you encounter native binding issues, see our [Windows Compatibility Guide](./WINDOWS_COMPATIBILITY.md).

### Basic Usage

#### Browser

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/dist/tf.min.js"></script>
<script src="https://unpkg.com/clustering-tfjs/dist/clustering.browser.js"></script>

<script>
async function demo() {
  // Initialize the library
  await ClusteringTFJS.Clustering.init({ backend: 'webgl' });
  
  // Use algorithms
  const kmeans = new ClusteringTFJS.KMeans({ nClusters: 3 });
  const data = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]];
  const labels = await kmeans.fitPredict(data);
  console.log(labels); // [0, 0, 1, 1, 0, 2]
}
demo();
</script>
```

#### Node.js

```typescript
import { Clustering } from 'clustering-tfjs';

// Initialize (optional - auto-detects best backend)
await Clustering.init();

// Use algorithms
const kmeans = new Clustering.KMeans({ nClusters: 3 });
const data = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]];
const labels = await kmeans.fitPredict(data);
console.log(labels); // [0, 0, 1, 1, 0, 2]
```

## Installation

### For Node.js

```bash
# Basic installation (pure JavaScript backend)
npm install clustering-tfjs

# Recommended: With native acceleration
npm install clustering-tfjs @tensorflow/tfjs-node

# Optional: With GPU support
npm install clustering-tfjs @tensorflow/tfjs-node-gpu
```

### For Browser

The browser bundle is available via CDN:

```html
<!-- Load TensorFlow.js -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/dist/tf.min.js"></script>

<!-- Load clustering-tfjs -->
<script src="https://unpkg.com/clustering-tfjs/dist/clustering.browser.js"></script>
```

Or install via npm and use with a bundler:

```bash
npm install clustering-tfjs @tensorflow/tfjs
```

## Algorithms

### K-Means Clustering

- Classic centroid-based clustering
- Supports custom initialization methods
- K-Means++ initialization by default

### Spectral Clustering

- Graph-based clustering using eigendecomposition
- Ideal for non-convex clusters
- Supports custom affinity functions

### Agglomerative Clustering

- Hierarchical bottom-up clustering
- Multiple linkage criteria (ward, complete, average, single)
- Memory efficient implementation

### Self-Organizing Maps (SOM)

- Neural network-based unsupervised learning
- Topology-preserving dimensionality reduction
- Supports rectangular and hexagonal grid topologies
- Multiple initialization methods (random, linear, PCA)
- Flexible neighborhood functions (gaussian, bubble, mexican_hat)
- Incremental/online learning support for streaming data
- Ideal for visualization and exploratory data analysis

## Validation Metrics

The library includes three validation metrics to evaluate clustering quality and optimize the number of clusters:

### Silhouette Score

Measures how similar an object is to its own cluster compared to other clusters. Range: [-1, 1], higher is better.

### Davies-Bouldin Index

Evaluates intra-cluster and inter-cluster distances. Range: [0, ∞), lower is better.

### Calinski-Harabasz Index

Ratio of between-cluster to within-cluster dispersion. Range: [0, ∞), higher is better.

### Finding Optimal Number of Clusters

The library includes a built-in `findOptimalClusters` function that automatically determines the optimal number of clusters:

```typescript
import { findOptimalClusters } from 'clustering-tfjs';

// Find optimal k between 2 and 10 clusters
const result = await findOptimalClusters(data, {
  minClusters: 2,
  maxClusters: 10,
  algorithm: 'kmeans'  // or 'spectral', 'agglomerative'
});

console.log(`Optimal number of clusters: ${result.optimal.k}`);
console.log(`Silhouette score: ${result.optimal.silhouette}`);
console.log(`All evaluations:`, result.evaluations);

// Advanced usage with custom scoring
const customResult = await findOptimalClusters(data, {
  maxClusters: 8,
  algorithm: 'spectral',
  algorithmParams: { affinity: 'nearest_neighbors' },
  metrics: ['silhouette', 'calinskiHarabasz'],  // Skip Davies-Bouldin
  scoringFunction: (evaluation) => evaluation.silhouette * 2 + evaluation.calinskiHarabasz
});
```

## Platform Detection & Backend Selection

The library automatically detects your environment and selects the best backend:

```typescript
import { Clustering } from 'clustering-tfjs';

// Check current platform
console.log('Platform:', Clustering.platform); // 'browser' or 'node'

// Check available features
console.log('Features:', Clustering.features);
// {
//   gpuAcceleration: true,
//   wasmSimd: false,
//   nodeBindings: true,
//   webgl: false
// }

// Manually select backend
await Clustering.init({ backend: 'webgl' }); // Browser
await Clustering.init({ backend: 'tensorflow' }); // Node.js
```

### Available Backends

| Backend | Environment | Use Case | Performance |
|---------|------------|----------|-------------|
| `cpu` | Both | Pure JS fallback | Baseline |
| `webgl` | Browser | GPU acceleration | 5-10x faster |
| `wasm` | Browser | CPU optimization | 2-3x faster |
| `tensorflow` | Node.js | Native bindings | 10-20x faster |

The library automatically selects the best available backend if not specified.

## API Reference

### Common Interface

All algorithms implement the same interface:

```typescript
interface ClusteringAlgorithm {
  fit(X: Tensor2D | number[][]): Promise<void>;
  predict(X: Tensor2D | number[][]): Promise<number[]>;
  fitPredict(X: Tensor2D | number[][]): Promise<number[]>;
}
```

### KMeans

```typescript
new KMeans({
  nClusters: number;
  init?: 'k-means++' | 'random' | number[][];
  nInit?: number;
  maxIter?: number;
  tol?: number;
  // backend selection coming in future version
})
```

### SpectralClustering

```typescript
new SpectralClustering({
  nClusters: number;
  affinity?: 'rbf' | 'nearest_neighbors';
  gamma?: number;
  nNeighbors?: number;
  // backend selection coming in future version
})
```

### AgglomerativeClustering

```typescript
new AgglomerativeClustering({
  nClusters: number;
  linkage?: 'ward' | 'complete' | 'average' | 'single';
  // backend selection coming in future version
})
```

### Validation Metrics

```typescript
// Silhouette Score: [-1, 1], higher is better
silhouetteScore(X: Tensor2D | number[][], labels: number[]): Promise<number>

// Davies-Bouldin Index: [0, ∞), lower is better  
daviesBouldin(X: Tensor2D | number[][], labels: number[]): Promise<number>

// Calinski-Harabasz Index: [0, ∞), higher is better
calinskiHarabasz(X: Tensor2D | number[][], labels: number[]): Promise<number>
```

## Examples

### Live Demos

Try these interactive examples directly in your browser:

- [**Interactive Clustering Visualization**](https://observablehq.com/@observablehq/clustering-tfjs-demo) - Explore all algorithms with different datasets
- [**Local Examples**](examples/observable/) - Run examples locally with HTML files

Observable notebooks coming soon! In the meantime, check out the [local examples](examples/observable/) which can be:
- Opened directly in your browser
- Served locally with `npm run serve:examples`
- Used as templates for your own visualizations

### Code Examples

Coming soon: Example notebooks and CodePen demos

## Performance

Based on our benchmarks:

- K-Means: 0.5ms - 200ms depending on dataset size
- Spectral: 10ms - 2s (includes eigendecomposition)
- Agglomerative: 5ms - 500ms

See [benchmarks/](benchmarks/) for detailed performance data.

## Migration from scikit-learn

```python
# scikit-learn
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(X)
```

```typescript
// clustering-js
import { KMeans } from 'clustering-tfjs';
const kmeans = new KMeans({ nClusters: 3 });
const labels = await kmeans.fitPredict(X);
```

### Scikit-learn Compatibility

This library has been extensively tested for numerical parity with scikit-learn. Our test suite includes:

- Step-by-step comparisons with sklearn implementations
- Identical results for standard datasets
- Matching behavior for edge cases

See [`tools/sklearn_comparison/`](tools/sklearn_comparison/) for detailed comparison scripts and [`test/`](test/) for parity tests.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project.

## License

MIT

---

**Note**: This library is under active development. APIs may change in future versions.
