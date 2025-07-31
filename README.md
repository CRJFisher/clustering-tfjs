# clustering-tfjs

[![npm version](https://badge.fury.io/js/clustering-tfjs.svg)](https://www.npmjs.com/package/clustering-tfjs)
[![Build Status](https://github.com/CRJFisher/clustering-js/workflows/CI/badge.svg)](https://github.com/CRJFisher/clustering-js/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/TypeScript-Ready-blue.svg)](https://www.typescriptlang.org/)

Native TypeScript implementation of clustering algorithms powered by TensorFlow.js

## Features

- ✅ Pure TypeScript/JavaScript (no Python required)
- ✅ Multiple clustering algorithms (K-Means, Spectral, Agglomerative)
- ✅ Powered by TensorFlow.js for performance
- ✅ Works in Node.js (browser support coming soon)
- ✅ TypeScript support with full type definitions
- ✅ GPU acceleration available with tfjs-node-gpu
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
# Standard installation (macOS/Linux)
npm install clustering-tfjs @tensorflow/tfjs-node

# Windows installation (recommended)
npm install clustering-tfjs @tensorflow/tfjs
```

> **Note**: For Windows users or if you encounter native binding issues, see our [Windows Compatibility Guide](./WINDOWS_COMPATIBILITY.md). The library works with either `@tensorflow/tfjs-node` (faster, requires native bindings) or `@tensorflow/tfjs` (pure JavaScript, universal compatibility).

### Basic Usage

```typescript
import { KMeans } from 'clustering-tfjs';

// Simple example
const kmeans = new KMeans({ nClusters: 3 });
const data = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]];
const labels = await kmeans.fitPredict(data);
console.log(labels); // [0, 0, 1, 1, 0, 2]
```

## Installation

### For Node.js

```bash
npm install clustering-js
```

The package includes `@tensorflow/tfjs-node` for CPU-optimized performance.

### For Browser (Coming Soon)

Browser support is planned for a future release. The library will automatically use WebGL or WASM backends when available.

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

## Backend Selection

| Backend | Use Case | Performance | Status |
|---------|----------|-------------|---------|
| tfjs-node (CPU) | Default Node.js backend | Baseline | ✅ Included |
| tfjs-node-gpu | Large datasets, ML pipelines | 5-20x faster | 🚧 Planned |
| WebGL | Browser GPU acceleration | TBD | 🚧 Planned |
| WASM | Browser CPU optimization | TBD | 🚧 Planned |

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
