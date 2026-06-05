# Source architecture

`clustering-tfjs` is organized **by domain** — each top-level folder under `src/`
groups the code for one capability, so the layout reads as a description of what
the library does. There is no catch-all `utils/` bucket.

## Module map

| Folder | Responsibility | Key modules |
| --- | --- | --- |
| `src/backend/` | TensorFlow.js access: lazy adapter, backend selection/initialization, platform detection, per-platform loaders. | `adapter.ts`, `backend.ts`, `platform.ts`, `platform_types.ts`, `loader.{browser,node,rn}.ts` |
| `src/clustering/` | The estimators and the public init namespace. | `kmeans.ts`, `spectral.ts`, `agglomerative.ts`, `som.ts`, `linkage.ts`, `spectral_consensus.ts`, `spectral_optimization.ts`, `som_neighborhood.ts`, `types.ts`, `init.ts` |
| `src/eigen/` | Eigendecomposition for spectral embedding. | `qr.ts`, `improved.ts`, `post.ts`, `lanczos.ts`, `orthogonalize.ts`, `constant_eigenvector.ts`, `smallest_eigenvectors_with_values.ts` |
| `src/graph/` | Graph construction for spectral clustering. | `affinity.ts`, `laplacian.ts`, `connected_components.ts`, `component_indicators.ts` |
| `src/distance/` | Pairwise distance computation. | `pairwise_distance.ts` |
| `src/model_selection/` | Choosing the number of clusters. | `find_optimal_clusters.ts`, `compute_wss.ts`, `kneedle.ts` |
| `src/validation/` | Clustering quality metrics. | `silhouette.ts`, `davies_bouldin.ts`, `calinski_harabasz.ts`, `adjusted_rand_index.ts`, `normalized_mutual_info.ts`, `contingency.ts`, `validate.ts` |
| `src/tensor/` | Tensor conversion helpers and runtime type guards. | `tensor_ops.ts`, `tensor_guards.ts` |
| `src/random/` | Deterministic random number generation. | `index.ts`, `mt19937.ts` |
| `src/datasets/` | Synthetic dataset generators. | `synthetic.ts` |
| `src/visualization/` | SOM visualization helpers. | `som_visualization.ts` |

`src/index.ts` is the single public entry point; it re-exports the estimators,
the `Clustering` init namespace, the validation metrics, and the model-selection
and distance helpers.

## Dependency direction

```
clustering ─┬─> eigen ──> tensor ──> backend
            ├─> graph ──> distance ──> backend
            ├─> model_selection ──> validation
            ├─> random
            └─> backend
```

`backend/` is the foundation: every other folder reaches TensorFlow.js through
`backend/adapter.ts`, never by importing `@tensorflow/*` directly. This keeps
backend selection (Node / browser / WASM / React Native) in one place.

## Conventions

- **Naming** is Python-style: `snake_case` for values and file names,
  `PascalCase` for classes/interfaces/type-aliases/enums, `UPPER_SNAKE_CASE` for
  constants.
- **Tests are colocated**: `foo.ts` is covered by `foo.test.ts` in the same
  folder. Shared reference fixtures (scikit-learn parity data) live under
  `__fixtures__/`.
- **Benchmarks** live in the top-level `benchmarks/` folder, outside `src/`, so
  they are never shipped in the published `dist/`.
