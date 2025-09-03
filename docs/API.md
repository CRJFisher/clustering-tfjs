# API Reference

Complete API documentation for clustering-tfjs.

## Table of Contents

- [Clustering Algorithms](#clustering-algorithms)
  - [KMeans](#kmeans)
  - [SpectralClustering](#spectralclustering)
  - [AgglomerativeClustering](#agglomerativeclustering)
  - [SOM](#som-self-organizing-maps)
- [Validation Metrics](#validation-metrics)
  - [silhouetteScore](#silhouettescore)
  - [daviesBouldin](#daviesbouldin)
  - [calinskiHarabasz](#calinskiharabasz)
- [Utility Functions](#utility-functions)
  - [findOptimalClusters](#findoptimalclusters)
- [Types](#types)

## Clustering Algorithms

All clustering algorithms implement the `BaseClustering` interface:

```typescript
interface BaseClustering<P extends BaseClusteringParams> {
  params: P;
  labels_: LabelVector | null;
  
  fit(X: DataMatrix): Promise<void>;
  predict(X: DataMatrix): Promise<LabelVector>;
  fitPredict(X: DataMatrix): Promise<LabelVector>;
}
```

### KMeans

K-means clustering using Lloyd's algorithm with K-means++ initialization.

#### Constructor

```typescript
new KMeans(params: KMeansParams)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nClusters` | `number` | required | Number of clusters to form |
| `init` | `'k-means++' \| 'random' \| number[][]` | `'k-means++'` | Initialization method |
| `nInit` | `number` | `10` | Number of initializations to run |
| `maxIter` | `number` | `300` | Maximum iterations per run |
| `tol` | `number` | `1e-4` | Convergence tolerance |
| `randomState` | `number` | `undefined` | Random seed for reproducibility |

#### Example

```typescript
import { KMeans } from 'clustering-tfjs';

const kmeans = new KMeans({
  nClusters: 3,
  init: 'k-means++',
  nInit: 10,
  maxIter: 300
});

const data = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]];
const labels = await kmeans.fitPredict(data);
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

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nClusters` | `number` | required | Number of clusters |
| `affinity` | `'rbf' \| 'nearest_neighbors'` | `'rbf'` | Affinity matrix construction method |
| `gamma` | `number` | `1.0` | Kernel coefficient for RBF |
| `nNeighbors` | `number` | `10` | Number of neighbors for k-NN |
| `nInit` | `number` | `10` | Number of K-means initializations |
| `randomState` | `number` | `undefined` | Random seed |

#### Example

```typescript
import { SpectralClustering } from 'clustering-tfjs';

const spectral = new SpectralClustering({
  nClusters: 2,
  affinity: 'nearest_neighbors',
  nNeighbors: 7
});

const labels = await spectral.fitPredict(data);
```

### AgglomerativeClustering

Hierarchical clustering using bottom-up approach.

#### Constructor

```typescript
new AgglomerativeClustering(params: AgglomerativeParams)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nClusters` | `number` | required | Number of clusters |
| `linkage` | `'ward' \| 'complete' \| 'average' \| 'single'` | `'ward'` | Linkage criterion |

#### Example

```typescript
import { AgglomerativeClustering } from 'clustering-tfjs';

const agglo = new AgglomerativeClustering({
  nClusters: 3,
  linkage: 'ward'
});

const labels = await agglo.fitPredict(data);
```

### SOM (Self-Organizing Maps)

Kohonen Self-Organizing Maps for unsupervised learning and visualization.

#### Constructor

```typescript
new SOM(params: SOMParams)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nClusters` | `number` | required | Total number of neurons (gridWidth × gridHeight) |
| `gridWidth` | `number` | required | Width of the SOM grid |
| `gridHeight` | `number` | required | Height of the SOM grid |
| `topology` | `'rectangular' \| 'hexagonal'` | `'rectangular'` | Grid topology (4/8 or 6 neighbors) |
| `neighborhood` | `'gaussian' \| 'bubble' \| 'mexican_hat'` | `'gaussian'` | Neighborhood function for weight updates |
| `initialization` | `'random' \| 'linear' \| 'pca'` | `'linear'` | Weight initialization method |
| `learningRate` | `number \| DecayFunction` | `0.5` | Initial learning rate or custom decay function |
| `radius` | `number \| DecayFunction` | `max(gridWidth, gridHeight) / 2` | Initial neighborhood radius or custom decay |
| `numEpochs` | `number` | `100` | Number of training epochs |
| `tol` | `number` | `1e-4` | Convergence tolerance |
| `randomState` | `number` | `undefined` | Random seed for reproducibility |

#### Methods

##### fitPredict(X: DataMatrix): Promise<LabelVector>
Train the SOM and return cluster assignments.

##### fit(X: DataMatrix): Promise<void>
Train the SOM on the provided data. Supports incremental/online learning - can be called multiple times with new data batches to continue training.

##### getWeights(): tf.Tensor3D
Get the trained weight vectors of all neurons. Shape: [gridHeight, gridWidth, nFeatures]

##### getUMatrix(): tf.Tensor2D
Calculate the U-matrix (unified distance matrix) showing average distances between neurons and their neighbors. Useful for visualization.

##### quantizationError(): number
Calculate the average distance between samples and their Best Matching Units (BMUs).

##### topographicError(X?: DataMatrix): Promise<number>
Calculate the proportion of samples for which the first and second BMUs are not adjacent. Lower values indicate better topology preservation.

#### Example

```typescript
import { SOM } from 'clustering-tfjs';

// Create a 5x5 SOM with hexagonal topology
const som = new SOM({
  gridWidth: 5,
  gridHeight: 5,
  nClusters: 25,
  topology: 'hexagonal',
  neighborhood: 'gaussian',
  initialization: 'pca',
  learningRate: 0.5,
  radius: 2.5,
  numEpochs: 100,
  randomState: 42
});

// Train on data
const data = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]];
const labels = await som.fitPredict(data);

// Get the trained weights
const weights = som.getWeights();
console.log('Weight shape:', weights.shape); // [5, 5, 2]

// Calculate U-matrix for visualization
const uMatrix = som.getUMatrix();
console.log('U-matrix shape:', uMatrix.shape); // [5, 5]

// Evaluate map quality
const quantError = som.quantizationError();
console.log('Quantization error:', quantError);

const topoError = await som.topographicError(data);
console.log('Topographic error:', topoError);

// Clean up tensors
weights.dispose();
uMatrix.dispose();
```

## Validation Metrics

### silhouetteScore

Computes the mean Silhouette Coefficient of all samples.

```typescript
function silhouetteScore(
  X: DataMatrix,
  labels: LabelVector
): Promise<number>
```

#### Parameters

- `X`: Input data (n_samples × n_features)
- `labels`: Cluster labels for each sample

#### Returns

Silhouette score (range: [-1, 1], higher is better)

#### Example

```typescript
import { KMeans, silhouetteScore } from 'clustering-tfjs';

const kmeans = new KMeans({ nClusters: 3 });
const labels = await kmeans.fitPredict(data);
const score = await silhouetteScore(data, labels);
console.log(`Silhouette score: ${score}`);
```

### daviesBouldin

Computes the Davies-Bouldin index.

```typescript
function daviesBouldin(
  X: DataMatrix,
  labels: LabelVector
): Promise<number>
```

#### Parameters

- `X`: Input data (n_samples × n_features)
- `labels`: Cluster labels for each sample

#### Returns

Davies-Bouldin index (range: [0, ∞), lower is better)

### calinskiHarabasz

Computes the Calinski-Harabasz index.

```typescript
function calinskiHarabasz(
  X: DataMatrix,
  labels: LabelVector
): Promise<number>
```

#### Parameters

- `X`: Input data (n_samples × n_features)
- `labels`: Cluster labels for each sample

#### Returns

Calinski-Harabasz index (range: [0, ∞), higher is better)

## Utility Functions

### findOptimalClusters

Automatically determines the optimal number of clusters for a dataset.

```typescript
function findOptimalClusters(
  X: DataMatrix,
  options?: FindOptimalClustersOptions
): Promise<{
  optimal: ClusterEvaluation;
  evaluations: ClusterEvaluation[];
}>
```

#### Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `minClusters` | `number` | `2` | Minimum clusters to test |
| `maxClusters` | `number` | `10` | Maximum clusters to test |
| `algorithm` | `'kmeans' \| 'spectral' \| 'agglomerative'` | `'kmeans'` | Algorithm to use |
| `algorithmParams` | `object` | `{}` | Algorithm-specific parameters |
| `metrics` | `string[]` | `['silhouette', 'daviesBouldin', 'calinskiHarabasz']` | Metrics to compute |
| `scoringFunction` | `(eval: ClusterEvaluation) => number` | Combined score | Custom scoring |

#### Returns

- `optimal`: Best clustering configuration
- `evaluations`: All tested configurations sorted by score

#### Example

```typescript
import { findOptimalClusters } from 'clustering-tfjs';

const result = await findOptimalClusters(data, {
  minClusters: 2,
  maxClusters: 8,
  algorithm: 'kmeans'
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
  daviesBouldin: number;
  calinskiHarabasz: number;
  combinedScore: number;
  labels: number[];
}
```

## Error Handling

All methods may throw errors for:

- Invalid parameters (e.g., `nClusters < 2`)
- Insufficient data (e.g., fewer samples than clusters)
- Numerical issues (e.g., singular matrices)

Always wrap calls in try-catch when handling user input:

```typescript
try {
  const labels = await kmeans.fitPredict(data);
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
  const labels = await kmeans.fitPredict(dataTensor);
  // Use labels
} finally {
  dataTensor.dispose();
}
```