# API Reference

Complete API documentation for clustering-tfjs.

## Table of Contents

- [Clustering Algorithms](#clustering-algorithms)
  - [KMeans](#kmeans)
  - [SpectralClustering](#spectralclustering)
  - [AgglomerativeClustering](#agglomerativeclustering)
  - [HDBSCAN](#hdbscan)
  - [SOM](#som-self-organizing-maps)
- [Cluster Representations](#cluster-representations)
  - [select_medoids](#select_medoids)
- [Cluster Tracking](#cluster-tracking)
  - [track_clusters](#track_clusters)
- [Decomposition](#decomposition)
  - [PCA](#pca)
- [Validation Metrics](#validation-metrics)
  - [Noise (-1) handling](#noise--1-handling)
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

| Parameter      | Type     | Default     | Description                                     |
| -------------- | -------- | ----------- | ----------------------------------------------- |
| `n_clusters`   | `number` | required    | Number of clusters to form                      |
| `n_init`       | `number` | `10`        | Number of initializations to run                |
| `max_iter`     | `number` | `300`       | Maximum iterations per run                      |
| `tol`          | `number` | `1e-4`      | Convergence tolerance                           |
| `metric`       | `string` | `euclidean` | `euclidean`, or `cosine` for spherical k-means  |
| `random_state` | `number` | `undefined` | Random seed for reproducibility                 |

With `metric: 'cosine'`, KMeans runs spherical k-means: rows are L2-normalized
onto the unit sphere and all distances are cosine distances.

#### Methods

KMeans supports inference on unseen data and JSON serialization:

- `predict(X: DataMatrix): Promise<number[]>` — assign each row of `X` to its nearest fitted centroid (cosine models L2-normalize first).
- `get_centroids(): number[][]` — the learned centroids as a plain array.
- `to_json(): KMeansJSON` / `static from_json(json): KMeans` — round-trip a fitted model (centroids, params, inertia). A restored model reproduces `predict` exactly without re-fitting.

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

// Access cluster centers as a plain number[][]
const centers = kmeans.get_centroids();

// Assign new, unseen points to the fitted clusters
const newLabels = await kmeans.predict([[2, 3]]);
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

#### Parameters

| Parameter    | Type                                            | Default  | Description        |
| ------------ | ----------------------------------------------- | -------- | ------------------ |
| `n_clusters` | `number`                                        | required | Number of clusters |
| `linkage`    | `'ward' \| 'complete' \| 'average' \| 'single'` | `'ward'` | Linkage criterion  |

#### Example

```typescript
import { AgglomerativeClustering } from 'clustering-tfjs';

const agglo = new AgglomerativeClustering({
  n_clusters: 3,
  linkage: 'ward',
});

const labels = await agglo.fit_predict(data);
```

#### Predict and serialization

SpectralClustering and AgglomerativeClustering are **transductive**: their
labels come from a graph embedding or a linkage hierarchy built over the
specific input set, with no centroid or parametric model that can assign an
unseen point. They therefore expose **no `predict`** and **no
`to_json`/`from_json`**. KMeans is parametric — its centroids fully determine
assignment — so it supports both `predict` and JSON serialization. This
asymmetry is intentional, not a gap.

### HDBSCAN

Hierarchical density-based clustering. Discovers clusters of varying density
without a preset cluster count and flags samples in sparse regions as **noise**
(`-1`). Fit-only — like AgglomerativeClustering there is no `predict`.

#### Constructor

```typescript
new HDBSCAN(params?: Partial<HDBSCANParams>)
```

#### Parameters

| Parameter                   | Type     | Default       | Description                                       |
| --------------------------- | -------- | ------------- | ------------------------------------------------- |
| `min_cluster_size`          | `number` | `5`           | Smallest admissible cluster; smaller groups become noise (min 2) |
| `min_samples`               | `number` | `min_cluster_size` | Core-distance neighbourhood size (the point counts as its own first neighbour) |
| `metric`                    | `string` | `euclidean`   | `euclidean`, `manhattan`, or `precomputed` (an `(n,n)` distance matrix; the cosine path supplies a cosine distance matrix) |
| `cluster_selection_epsilon` | `number` | `0`           | Merge clusters born below the `1/epsilon` density level |
| `cluster_selection_method`  | `string` | `eom`         | `eom` (Excess of Mass) or `leaf`                  |
| `store_exemplars`           | `boolean`| `false`       | Populate `exemplar_indices_` (most-persistent point per cluster) |

#### Fitted attributes

- `labels_: number[] | null` — cluster ids `>= 0`, `-1` for noise.
- `probabilities_: number[] | null` — per-sample membership strength in `[0, 1]`.
- `exemplar_indices_: Map<number, number> | null` — exemplar sample index per cluster label (when `store_exemplars`).

#### Example

```typescript
import { HDBSCAN } from 'clustering-tfjs';

const hdbscan = new HDBSCAN({ min_cluster_size: 5 });
const labels = await hdbscan.fit_predict(data); // e.g. [0, 0, -1, 1, 1, -1]
console.log(hdbscan.probabilities_);
```

For cosine geometry, pass a precomputed cosine distance matrix with
`metric: 'precomputed'` (scikit-learn parity uses the same route).

> **Parity note.** HDBSCAN labels and probabilities match scikit-learn closely
> but not bit-for-bit: mutual-reachability weight ties are ordered differently
> across implementations (numpy's unstable `argsort`), shifting a few boundary
> points. The condensed-tree and Excess-of-Mass core is validated for exact
> parity against scikit-learn's own single-linkage hierarchy.

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

## Cluster Representations

Every estimator can surface a representative vector per cluster through one
`ClusterRepresentations` shape, so downstream code (summarization, labelling,
nearest-representative lookups) works the same regardless of algorithm:

- **KMeans** exposes synthetic `centroids_` (and `get_centroids()`).
- **AgglomerativeClustering** and **SpectralClustering** expose `medoid_indices_`
  via `compute_medoids(X)` — the real sample closest to each cluster's mean.
- **HDBSCAN** exposes `exemplar_indices_` (most-persistent point per cluster).

```typescript
interface ClusterRepresentations {
  centroids_?: tf.Tensor2D | null;
  medoid_indices_?: Int32Array | null;
  exemplar_indices_?: Map<number, number> | null;
}
```

### select_medoids

```typescript
select_medoids(
  X: DataMatrix,
  labels: LabelVector,
  n_clusters: number,
  metric?: ClusteringMetric, // 'euclidean' | 'manhattan' | 'cosine'
): Promise<{ indices: Int32Array; distances: Float32Array }>
```

For each cluster, returns the index of the sample closest to that cluster's mean
under the requested metric (the medoid), plus its distance. Runs in `O(n*d)`
without materialising an `n×n` matrix. Noise (`-1`) labels are ignored; a cluster
with no members yields index `-1`.

```typescript
const agglo = new AgglomerativeClustering({ n_clusters: 3, metric: 'cosine' });
await agglo.fit(data);
const medoids = await agglo.compute_medoids(data); // Int32Array, one index per cluster
```

## Cluster Tracking

### track_clusters

```typescript
track_clusters(
  prev: number[][],          // representative vectors of previous snapshot's clusters
  curr: number[][],          // representative vectors of current snapshot's clusters
  options?: { threshold?: number },  // similarity threshold, default 0.5
  prev_state?: TrackingState,        // thread between frames for stable lifeline ids
): TrackingResult
```

Matches clusters across two consecutive snapshots by cosine similarity of their
representative vectors (e.g. KMeans centroids or HDBSCAN exemplars), via an
optimal bipartite (Hungarian) assignment with pruning of pairs below
`threshold`. Each cluster is classified as `PERSIST`, `EMERGE`, `DIE`, `MERGE`,
or `SPLIT`, with a stable lifeline id carried forward. The function is
**stateless**: the caller owns and threads the returned `state`.

```typescript
const r1 = track_clusters(prevCentroids, currCentroids, { threshold: 0.6 });
console.log(r1.transitions);   // [{ type: 'PERSIST', prev: [0], curr: [1], lifeline_id: 0 }, ...]
const r2 = track_clusters(currCentroids, nextCentroids, { threshold: 0.6 }, r1.state);
```

Rectangular cases (differing cluster counts) are handled — extra clusters on
either side surface as `EMERGE`/`DIE`.

## Decomposition

### PCA

Principal Component Analysis matching `sklearn.decomposition.PCA(svd_solver='full')`
up to per-component sign. Components are found by power iteration with deflation
(the TensorFlow.js backend has no eigendecomposition).

```typescript
new PCA({ n_components: number; random_state?: number })
```

| Method                       | Description                                                  |
| ---------------------------- | ------------------------------------------------------------ |
| `fit(X)`                     | Compute `components_`, `explained_variance_`, `mean_`        |
| `transform(X)`               | Project into component space (`n × n_components`)            |
| `fit_transform(X)`           | `fit` then `transform`                                       |
| `inverse_transform(Z)`       | Reconstruct from component space                             |
| `to_json()` / `from_json()`  | Round-trip a fitted estimator                               |

`fit` throws if `n_components` exceeds the number of features; `fit_predict`
throws (PCA is a reducer, not a clusterer).

```typescript
import { PCA } from 'clustering-tfjs';

const pca = new PCA({ n_components: 2, random_state: 0 });
const reduced = pca.fit_transform(highDimData);
// Pre-project before density clustering:
const labels = await new HDBSCAN({ min_cluster_size: 5 }).fit_predict(reduced);
```

## Validation Metrics

### Noise (-1) handling

Cluster labels carry one library-wide meaning (see
`backlog/decisions/decision-1`): non-density estimators emit dense labels
`0..n_clusters-1`; density estimators (HDBSCAN) emit `-1` for **noise** — samples
in no cluster. Internal-validation metrics measure genuine clusters, so
`silhouette_*`, `davies_bouldin*`, and `calinski_harabasz*` **exclude `-1`
samples before computing any distance or dispersion**. They stay well-defined at
the degenerate boundaries this introduces: all-noise input, or one cluster plus
noise, return a defined `0` with no division by zero. A genuine fewer-than-two-
clusters input with **no** noise still throws.

`silhouette_score` and `davies_bouldin` also accept a `metric` argument
(`'euclidean' | 'cosine'`); `calinski_harabasz` is variance-based and
metric-independent.

### silhouette_score

Computes the mean Silhouette Coefficient of all samples (noise excluded).

```typescript
function silhouette_score(
  X: DataMatrix,
  labels: LabelVector,
  metric?: 'euclidean' | 'cosine',
): number;
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

| Parameter          | Type                                        | Default                                                 | Description                   |
| ------------------ | ------------------------------------------- | ------------------------------------------------------- | ----------------------------- |
| `min_clusters`     | `number`                                    | `2`                                                     | Minimum clusters to test      |
| `max_clusters`     | `number`                                    | `10`                                                    | Maximum clusters to test      |
| `algorithm`        | `'kmeans' \| 'spectral' \| 'agglomerative'` | `'kmeans'`                                              | Algorithm to use              |
| `algorithm_params` | `object`                                    | `{}`                                                    | Algorithm-specific parameters |
| `metrics`          | `string[]`                                  | `['silhouette', 'davies_bouldin', 'calinski_harabasz']` | Metrics to compute            |
| `scoring_function` | `(eval: ClusterEvaluation) => number`       | Combined score                                          | Custom scoring                |

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
