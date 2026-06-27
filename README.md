# clustering-tfjs

[![npm version](https://badge.fury.io/js/clustering-tfjs.svg)](https://www.npmjs.com/package/clustering-tfjs)
[![Build Status](https://github.com/CRJFisher/clustering-tfjs/workflows/CI/badge.svg)](https://github.com/CRJFisher/clustering-tfjs/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/TypeScript-Ready-blue.svg)](https://www.typescriptlang.org/)

**scikit-learn clustering, GPU-accelerated, 100% in your browser — no Python, no install.**

[![WebGPU races CPU on the same seeded dataset; the GPU lane finishes far ahead on Spectral RBF affinity at n = 10,000.](https://CRJFisher.github.io/clustering-tfjs/race-reference.svg)](https://CRJFisher.github.io/clustering-tfjs/)

## ▶ [Open the live demo (no install) →](https://CRJFisher.github.io/clustering-tfjs/)

Watch WebGPU race CPU on the same dataset, then recreate the scikit-learn toy-dataset grid across all five algorithms — live, on **your** hardware.

## 30-second quickstart

```bash
npm install clustering-tfjs
```

```typescript
import { Clustering, KMeans } from 'clustering-tfjs';

await Clustering.init(); // auto-detects the best available backend

const model = new KMeans({ n_clusters: 3 });
const labels = await model.fit_predict([
  [1, 2],
  [1.5, 1.8],
  [5, 8],
  [8, 8],
  [1, 0.6],
  [9, 11],
]);
console.log(labels); // [0, 0, 1, 1, 0, 2]
```

## Table of Contents

1. [Quick Start](#quick-start)
2. [Features](#features)
3. [Installation](#installation)
4. [Algorithms](#algorithms)
5. [Validation Metrics](#validation-metrics)
6. [Backend Selection](#backend-selection)
7. [API Reference](#api-reference)
8. [Examples](#examples)
9. [Performance](#performance)
10. [Migration from scikit-learn](#migration-from-scikit-learn)
11. [Contributing](#contributing)
12. [License](#license)

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
    const kmeans = new ClusteringTFJS.KMeans({ n_clusters: 3 });
    const data = [
      [1, 2],
      [1.5, 1.8],
      [5, 8],
      [8, 8],
      [1, 0.6],
      [9, 11],
    ];
    const labels = await kmeans.fit_predict(data);
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
const kmeans = new Clustering.KMeans({ n_clusters: 3 });
const data = [
  [1, 2],
  [1.5, 1.8],
  [5, 8],
  [8, 8],
  [1, 0.6],
  [9, 11],
];
const labels = await kmeans.fit_predict(data);
console.log(labels); // [0, 0, 1, 1, 0, 2]
```

## Features

- ✅ Pure TypeScript/JavaScript (no Python required)
- ✅ Five clustering algorithms (K-Means, Spectral, Agglomerative, HDBSCAN, SOM)
- ✅ GPU-accelerated with TensorFlow.js — WebGPU and WebGL in the browser, CUDA in Node.js
- ✅ **Works in both Node.js and browsers**
- ✅ Automatic backend selection with graceful WebGPU → WebGL → WASM → CPU fallback
- ✅ Platform-optimized bundles (49KB for browser, 163KB for Node.js)
- ✅ TypeScript support with full type definitions
- ✅ Extensively tested for parity with scikit-learn

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

### HDBSCAN

- Hierarchical density-based clustering with automatic cluster count
- Robust to noise — points that do not belong to any cluster are labeled `-1`
- Supports `euclidean`, `manhattan`, and `precomputed` distance metrics
- Two cluster selection methods: `eom` (excess of mass, default) and `leaf`

## Validation Metrics

The library includes three validation metrics to evaluate clustering quality and optimize the number of clusters:

### Silhouette Score

Measures how similar an object is to its own cluster compared to other clusters. Range: [-1, 1], higher is better.

### Davies-Bouldin Index

Evaluates intra-cluster and inter-cluster distances. Range: [0, ∞), lower is better.

### Calinski-Harabasz Index

Ratio of between-cluster to within-cluster dispersion. Range: [0, ∞), higher is better.

### Finding Optimal Number of Clusters

The library includes a built-in `find_optimal_clusters` function that automatically determines the optimal number of clusters:

```typescript
import { find_optimal_clusters } from 'clustering-tfjs';

// Find optimal k between 2 and 10 clusters
const result = await find_optimal_clusters(data, {
  min_clusters: 2,
  max_clusters: 10,
  algorithm: 'kmeans', // or 'spectral', 'agglomerative', 'som'
});

console.log(`Optimal number of clusters: ${result.optimal.k}`);
console.log(`Silhouette score: ${result.optimal.silhouette}`);
console.log(`All evaluations:`, result.evaluations);

// Advanced usage with custom scoring
const custom_result = await find_optimal_clusters(data, {
  max_clusters: 8,
  algorithm: 'spectral',
  algorithm_params: { affinity: 'nearest_neighbors' },
  metrics: ['silhouette', 'calinski_harabasz'], // Skip Davies-Bouldin
  scoring_function: (evaluation) =>
    evaluation.silhouette * 2 + evaluation.calinski_harabasz,
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
//   gpu_acceleration: true,
//   wasm_simd: false,
//   node_bindings: true,
//   webgl: false
// }

// Manually select backend
await Clustering.init({ backend: 'webgl' }); // Browser
await Clustering.init({ backend: 'tensorflow' }); // Node.js
```

### Available Backends

| Backend      | Environment | Use Case         | Performance   |
| ------------ | ----------- | ---------------- | ------------- |
| `cpu`        | Both        | Pure JS fallback | Baseline      |
| `webgl`      | Browser     | GPU acceleration | 5-10x faster  |
| `wasm`       | Browser     | CPU optimization | 2-3x faster   |
| `tensorflow` | Node.js     | Native bindings  | 10-20x faster |

The library automatically selects the best available backend if not specified.

## API Reference

### Common Interface

All algorithms implement the same interface:

```typescript
interface ClusteringAlgorithm {
  fit(X: Tensor2D | number[][]): Promise<void>;
  fit_predict(X: Tensor2D | number[][]): Promise<number[]>;
}
```

### KMeans

```typescript
new KMeans({
  n_clusters: number;
  n_init?: number;
  max_iter?: number;
  tol?: number;
  random_state?: number;
})
```

### SpectralClustering

```typescript
new SpectralClustering({
  n_clusters: number;
  affinity?: 'rbf' | 'nearest_neighbors' | 'precomputed';
  gamma?: number;
  n_neighbors?: number;
})
```

`affinity: 'nearest_neighbors'` uses a sparse kNN connectivity graph, sparse
normalized-Laplacian operator, and matrix-free Lanczos eigensolver. This keeps
peak graph memory proportional to `n_samples * n_neighbors` and mirrors
scikit-learn's nearest-neighbor spectral clustering symmetrization. `rbf`,
`precomputed`, and callable affinities remain dense paths.

### AgglomerativeClustering

```typescript
new AgglomerativeClustering({
  // Provide exactly one stopping criterion:
  n_clusters?: number;
  distance_threshold?: number;
  linkage?: 'ward' | 'complete' | 'average' | 'single';
  metric?: 'euclidean' | 'manhattan' | 'cosine' | 'precomputed';
})
```

After `fit`, the estimator exposes `children_`, `distances_` (merge heights, aligned with `children_`), and `n_leaves_`. Use `metric: 'precomputed'` to pass a square, symmetric, zero-diagonal distance matrix directly (not allowed with `linkage: 'ward'`).

### SOM (Self-Organizing Maps)

```typescript
new SOM({
  grid_width: number;
  grid_height: number;
  topology?: 'rectangular' | 'hexagonal';
  neighborhood?: 'gaussian' | 'bubble' | 'mexican_hat';
  initialization?: 'random' | 'linear' | 'pca';
  learning_rate?: number | DecayFunction;
  radius?: number | DecayFunction;
  num_epochs?: number;
  tol?: number;
  random_state?: number;
})
```

Note: SOM additionally provides `predict()` and `partial_fit()` methods for labeling new data and online learning.

### HDBSCAN

```typescript
new HDBSCAN({
  min_cluster_size?: number;          // integer >= 2, default 5
  min_samples?: number;               // integer >= 1, default = min_cluster_size
  metric?: 'euclidean' | 'manhattan' | 'precomputed';  // default 'euclidean'
  cluster_selection_method?: 'eom' | 'leaf';           // default 'eom'
  cluster_selection_epsilon?: number; // >= 0, default 0
})
```

HDBSCAN determines the number of clusters automatically from the data. Points that do not belong to any cluster are assigned label `-1` (noise). After fitting, `labels_` exposes per-point cluster assignments and `probabilities_` exposes per-point cluster membership strength (0 for noise). Use `metric: 'precomputed'` to supply a square, symmetric, zero-diagonal distance matrix directly. Because HDBSCAN determines its own cluster count, it is not a valid `algorithm` for `find_optimal_clusters`.

### Validation Metrics

```typescript
// Silhouette Score: [-1, 1], higher is better
silhouette_score(X: Tensor2D | number[][], labels: number[]): Promise<number>

// Davies-Bouldin Index: [0, ∞), lower is better
davies_bouldin(X: Tensor2D | number[][], labels: number[]): Promise<number>

// Calinski-Harabasz Index: [0, ∞), higher is better
calinski_harabasz(X: Tensor2D | number[][], labels: number[]): Promise<number>
```

## Examples

### Live Demos

Try these interactive examples directly in your browser:

- [**Interactive Clustering Visualization**](https://observablehq.com/@observablehq/clustering-tfjs-demo) - Explore all algorithms with different datasets
- [**Local Examples**](examples/observable/) - Run examples locally with HTML files

Check out the [local examples](examples/observable/) which can be:

- Opened directly in your browser
- Served locally with `npm run serve:examples`
- Used as templates for your own visualizations

## Performance

Based on our benchmarks:

- K-Means: 0.5ms - 200ms depending on dataset size
- Spectral: 10ms - 2s (includes eigendecomposition)
- Spectral nearest-neighbors: sparse graph memory scales with `n_neighbors`,
  making large sample counts feasible when dense RBF affinity would be O(n²)
- Agglomerative: 5ms - 500ms
- SOM: training time scales with grid size and number of epochs
- HDBSCAN: dominated by mutual reachability distance computation, O(n²) for euclidean

See [benchmarks/](benchmarks/) for detailed performance data.

## Migration from scikit-learn

```python
# scikit-learn
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(X)
```

```typescript
// clustering-tfjs
import { KMeans } from 'clustering-tfjs';
const kmeans = new KMeans({ n_clusters: 3 });
const labels = await kmeans.fit_predict(X);
```

### Scikit-learn Compatibility

This library has been extensively tested for numerical parity with scikit-learn. Our test suite includes:

- Step-by-step comparisons with sklearn implementations
- Identical results for standard datasets
- Matching behavior for edge cases

See [`tools/sklearn_comparison/`](tools/sklearn_comparison/) for detailed comparison scripts. Parity tests are colocated with the source they cover (for example `src/clustering/spectral_reference.test.ts`).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project.

## License

MIT
