# API Reference

Complete API documentation for clustering-tfjs.

## Table of Contents

- [Clustering Algorithms](#clustering-algorithms)
  - [KMeans](#kmeans)
  - [SpectralClustering](#spectralclustering)
  - [AgglomerativeClustering](#agglomerativeclustering)
  - [SOM](#som-self-organizing-maps)
- [Validation Metrics](#validation-metrics)
  - [silhouette_score](#silhouettescore)
  - [davies_bouldin](#daviesbouldin)
  - [calinski_harabasz](#calinskiharabasz)
- [Utility Functions](#utility-functions)
  - [find_optimal_clusters](#findoptimalclusters)
- [Types](#types)

## Clustering Algorithms

All clustering algorithms implement the `BaseClustering` interface:

```typescript
interface BaseClustering<P extends CoreClusteringParams> {
  params: P;
  labels_: number[] | null;

  fit(X: DataMatrix): Promise<void>;
  fit_predict(X: DataMatrix): Promise<number[]>;
}
```

### KMeans

K-means clustering using Lloyd's algorithm with K-means++ initialization.

#### Constructor

```typescript
new KMeans(params: KMeansParams)
```

#### Parameters

| Parameter      | Type     | Default     | Description                      |
| -------------- | -------- | ----------- | -------------------------------- |
| `n_clusters`   | `number` | required    | Number of clusters to form       |
| `n_init`       | `number` | `10`        | Number of initializations to run |
| `max_iter`     | `number` | `300`       | Maximum iterations per run       |
| `tol`          | `number` | `1e-4`      | Convergence tolerance            |
| `random_state` | `number` | `undefined` | Random seed for reproducibility  |

#### Example

```typescript
import { KMeans } from 'clustering-tfjs';

const kmeans = new KMeans({
  n_clusters: 3,
  n_init: 10,
  max_iter: 300,
});

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

// Access cluster centers
const centers = kmeans.centroids_;
```

### SpectralClustering

Spectral clustering using graph Laplacian eigendecomposition.

#### Constructor

```typescript
new SpectralClustering(params: SpectralClusteringParams)
```

#### Parameters

| Parameter      | Type                           | Default     | Description                         |
| -------------- | ------------------------------ | ----------- | ----------------------------------- |
| `n_clusters`   | `number`                       | required    | Number of clusters                  |
| `affinity`     | `'rbf' \| 'nearest_neighbors'` | `'rbf'`     | Affinity matrix construction method |
| `gamma`        | `number`                       | `1.0`       | Kernel coefficient for RBF          |
| `n_neighbors`  | `number`                       | `10`        | Number of neighbors for k-NN        |
| `n_init`       | `number`                       | `10`        | Number of K-means initializations   |
| `random_state` | `number`                       | `undefined` | Random seed                         |

#### Example

```typescript
import { SpectralClustering } from 'clustering-tfjs';

const spectral = new SpectralClustering({
  n_clusters: 2,
  affinity: 'nearest_neighbors',
  n_neighbors: 7,
});

const labels = await spectral.fit_predict(data);
```

### AgglomerativeClustering

Hierarchical clustering using bottom-up approach.

#### Constructor

```typescript
new AgglomerativeClustering(params: AgglomerativeClusteringParams)
```

Provide **exactly one** of `n_clusters` or `distance_threshold` as the stopping criterion.

#### Parameters

| Parameter            | Type                                                      | Default       | Description                                                                                            |
| -------------------- | --------------------------------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------ |
| `n_clusters`         | `number`                                                  | —             | Number of clusters. Mutually exclusive with `distance_threshold`.                                      |
| `distance_threshold` | `number`                                                  | —             | Linkage distance at or above which clusters are not merged. Mutually exclusive with `n_clusters`.      |
| `linkage`            | `'ward' \| 'complete' \| 'average' \| 'single'`           | `'ward'`      | Linkage criterion.                                                                                     |
| `metric`             | `'euclidean' \| 'manhattan' \| 'cosine' \| 'precomputed'` | `'euclidean'` | Distance metric. `'precomputed'` accepts a distance matrix directly and is incompatible with `'ward'`. |

#### Properties (populated after `fit`)

| Property     | Type         | Description                                                                                         |
| ------------ | ------------ | --------------------------------------------------------------------------------------------------- |
| `labels_`    | `number[]`   | Cluster label per sample.                                                                           |
| `children_`  | `number[][]` | The two cluster ids merged at each step (sklearn convention; ids `0..n-1` are leaves).              |
| `distances_` | `number[]`   | Distance at which each merge occurred, aligned 1:1 with `children_`. Useful for dendrogram cutting. |
| `n_leaves_`  | `number`     | Number of leaves (input samples).                                                                   |

#### Example

```typescript
import { AgglomerativeClustering } from 'clustering-tfjs';

const agglo = new AgglomerativeClustering({
  n_clusters: 3,
  linkage: 'ward',
});

const labels = await agglo.fit_predict(data);
console.log(agglo.distances_); // merge heights for a dendrogram

// Stop by distance instead of a fixed cluster count:
const by_threshold = new AgglomerativeClustering({
  distance_threshold: 2.5,
  linkage: 'average',
});
await by_threshold.fit_predict(data);

// Cluster a precomputed (square, symmetric, zero-diagonal) distance matrix:
const precomputed = new AgglomerativeClustering({
  n_clusters: 3,
  linkage: 'average',
  metric: 'precomputed',
});
await precomputed.fit_predict(distance_matrix);
```

### SOM (Self-Organizing Maps)

Kohonen Self-Organizing Maps for unsupervised learning and visualization.

#### Constructor

```typescript
new SOM(params: SOMParams)
```

#### Parameters

| Parameter        | Type                                      | Default                            | Description                                    |
| ---------------- | ----------------------------------------- | ---------------------------------- | ---------------------------------------------- |
| `grid_width`     | `number`                                  | required                           | Width of the SOM grid                          |
| `grid_height`    | `number`                                  | required                           | Height of the SOM grid                         |
| `topology`       | `'rectangular' \| 'hexagonal'`            | `'rectangular'`                    | Grid topology (4/8 or 6 neighbors)             |
| `neighborhood`   | `'gaussian' \| 'bubble' \| 'mexican_hat'` | `'gaussian'`                       | Neighborhood function for weight updates       |
| `initialization` | `'random' \| 'linear' \| 'pca'`           | `'linear'`                         | Weight initialization method                   |
| `learning_rate`  | `number \| DecayFunction`                 | `0.5`                              | Initial learning rate or custom decay function |
| `radius`         | `number \| DecayFunction`                 | `max(grid_width, grid_height) / 2` | Initial neighborhood radius or custom decay    |
| `num_epochs`     | `number`                                  | `100`                              | Number of training epochs                      |
| `tol`            | `number`                                  | `1e-4`                             | Convergence tolerance                          |
| `random_state`   | `number`                                  | `undefined`                        | Random seed for reproducibility                |

#### Methods

##### fit_predict(X: DataMatrix): Promise<number[]>

Train the SOM and return cluster assignments.

##### fit(X: DataMatrix): Promise<void>

Train the SOM on the provided data. Supports incremental/online learning - can be called multiple times with new data batches to continue training.

##### cluster(n_clusters: number, options?: SOMClusterOptions): Promise<number[]>

Perform 2-phase clustering: agglomerative clustering on SOM weight vectors to produce `n_clusters` meaningful clusters. Returns one label per data point from the most recent `fit()` call.

##### getWeights(): number[][][]

Get the trained weight vectors of all neurons as a plain JavaScript array. Shape: [grid_height][grid_width][n_features]. Returns a snapshot — safe to use after `dispose()`.

##### getUMatrix(): tf.Tensor2D

Calculate the U-matrix (unified distance matrix) showing average distances between neurons and their neighbors. Useful for visualization. Caller owns the returned tensor.

##### quantizationError(): number

Calculate the average distance between samples and their Best Matching Units (BMUs).

##### topographicError(X?: DataMatrix): Promise<number>

Calculate the proportion of samples for which the first and second BMUs are not adjacent. Lower values indicate better topology preservation.

##### partial_fit(X: DataMatrix): Promise<void>

Incremental learning. Requires `online_mode: true`. Input must have the same number of features as the initial fit.

##### dispose(): void

Release all GPU/WebGL memory. Safe to call multiple times. Previously returned `getWeights()` arrays remain valid. Previously returned tensors from `getUMatrix()` are unaffected (caller-owned).

#### Example

```typescript
import { SOM } from 'clustering-tfjs';

// Create a 5x5 SOM with hexagonal topology
const som = new SOM({
  grid_width: 5,
  grid_height: 5,
  topology: 'hexagonal',
  neighborhood: 'gaussian',
  initialization: 'pca',
  learning_rate: 0.5,
  radius: 2.5,
  num_epochs: 100,
  random_state: 42,
});

// Train on data
const data = [
  [1, 2],
  [1.5, 1.8],
  [5, 8],
  [8, 8],
  [1, 0.6],
  [9, 11],
];
const labels = await som.fit_predict(data);

// Get meaningful clusters (2-phase: SOM + agglomerative)
const clusterLabels = await som.cluster(3);
console.log('Cluster labels:', clusterLabels); // e.g. [0, 0, 1, 2, 0, 2]

// Get the trained weights (plain array, no dispose needed)
const weights = som.getWeights();
console.log('Weight shape:', [
  weights.length,
  weights[0].length,
  weights[0][0].length,
]); // [5, 5, 2]

// Calculate U-matrix for visualization
const u_matrix = som.getUMatrix();
console.log('U-matrix shape:', u_matrix.shape); // [5, 5]

// Evaluate map quality
const quantError = som.quantizationError();
console.log('Quantization error:', quantError);

const topoError = await som.topographicError(data);
console.log('Topographic error:', topoError);

// Clean up — only tensor values need disposing
u_matrix.dispose();
som.dispose();
```

## Validation Metrics

### silhouette_score

Computes the mean Silhouette Coefficient of all samples.

```typescript
function silhouette_score(X: DataMatrix, labels: LabelVector): Promise<number>;
```

#### Parameters

- `X`: Input data (n_samples × n_features)
- `labels`: Cluster labels for each sample

#### Returns

Silhouette score (range: [-1, 1], higher is better)

#### Example

```typescript
import { KMeans, silhouette_score } from 'clustering-tfjs';

const kmeans = new KMeans({ n_clusters: 3 });
const labels = await kmeans.fit_predict(data);
const score = await silhouette_score(data, labels);
console.log(`Silhouette score: ${score}`);
```

### davies_bouldin

Computes the Davies-Bouldin index.

```typescript
function davies_bouldin(X: DataMatrix, labels: LabelVector): Promise<number>;
```

#### Parameters

- `X`: Input data (n_samples × n_features)
- `labels`: Cluster labels for each sample

#### Returns

Davies-Bouldin index (range: [0, ∞), lower is better)

### calinski_harabasz

Computes the Calinski-Harabasz index.

```typescript
function calinski_harabasz(X: DataMatrix, labels: LabelVector): Promise<number>;
```

#### Parameters

- `X`: Input data (n_samples × n_features)
- `labels`: Cluster labels for each sample

#### Returns

Calinski-Harabasz index (range: [0, ∞), higher is better)

## Utility Functions

### find_optimal_clusters

Automatically determines the optimal number of clusters for a dataset.

```typescript
function find_optimal_clusters(
  X: DataMatrix,
  options?: FindOptimalClustersOptions,
): Promise<{
  optimal: ClusterEvaluation;
  evaluations: ClusterEvaluation[];
}>;
```

#### Options

| Parameter          | Type                                                 | Default                                                 | Description                   |
| ------------------ | ---------------------------------------------------- | ------------------------------------------------------- | ----------------------------- |
| `min_clusters`     | `number`                                             | `2`                                                     | Minimum clusters to test      |
| `max_clusters`     | `number`                                             | `10`                                                    | Maximum clusters to test      |
| `algorithm`        | `'kmeans' \| 'spectral' \| 'agglomerative' \| 'som'` | `'kmeans'`                                              | Algorithm to use              |
| `algorithm_params` | `object`                                             | `{}`                                                    | Algorithm-specific parameters |
| `metrics`          | `string[]`                                           | `['silhouette', 'davies_bouldin', 'calinski_harabasz']` | Metrics to compute            |
| `scoring_function` | `(eval: ClusterEvaluation) => number`                | Combined score                                          | Custom scoring                |
| `method`           | `'combined' \| 'elbow' \| 'silhouette'`              | `'combined'`                                            | Selection method              |

With `algorithm: 'som'`, a single map is trained once (grid sized from the data) and each candidate `k` is produced by two-phase clustering — agglomerative grouping of the trained neuron weight vectors into `k` macro-clusters, then mapping each sample to its neuron's group. This means the number of clusters tracks `k`, not the SOM grid size.

#### Returns

- `optimal`: Best clustering configuration
- `evaluations`: All tested configurations sorted by score

#### Example

```typescript
import { find_optimal_clusters } from 'clustering-tfjs';

const result = await find_optimal_clusters(data, {
  min_clusters: 2,
  max_clusters: 8,
  algorithm: 'kmeans',
});

console.log(`Optimal clusters: ${result.optimal.k}`);
console.log(`Best silhouette: ${result.optimal.silhouette}`);
```

## Types

### DataMatrix

Input data type accepted by all algorithms:

```typescript
type DataMatrix = tf.Tensor2D | number[][];
```

### LabelVector

Cluster labels returned by algorithms:

```typescript
type LabelVector = tf.Tensor1D | number[];
```

### ClusterEvaluation

Result from cluster evaluation:

```typescript
interface ClusterEvaluation {
  k: number;
  silhouette: number;
  davies_bouldin: number;
  calinski_harabasz: number;
  combinedScore: number;
  labels: number[];
}
```

## Error Handling

All methods may throw errors for:

- Invalid parameters (e.g., `n_clusters < 2`)
- Insufficient data (e.g., fewer samples than clusters)
- Numerical issues (e.g., singular matrices)

Always wrap calls in try-catch when handling user input:

```typescript
try {
  const labels = await kmeans.fit_predict(data);
} catch (error) {
  console.error('Clustering failed:', error.message);
}
```

## Memory Management

When working with tensors directly:

1. Dispose tensors after use
2. Use `tf.tidy()` for automatic cleanup
3. The library handles internal tensor disposal

```typescript
const dataTensor = tf.tensor2d(data);
try {
  const labels = await kmeans.fit_predict(dataTensor);
  // Use labels
} finally {
  dataTensor.dispose();
}
```
