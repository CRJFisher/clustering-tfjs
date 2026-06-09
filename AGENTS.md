# Rules

## Naming conventions (Python-style)

- `snake_case` for variables, functions, methods, properties, parameters, and **file names**.
- `PascalCase` for classes, interfaces, type aliases, and enums.
- `UPPER_SNAKE_CASE` for module-level constants.
- No `camelCase` anywhere.

## Source layout

The library is organized by **what the code does** (action/domain folders), not by generic buckets like `utils/`:

- `src/backend/` — TensorFlow.js adapter, backend selection, platform detection and loaders.
- `src/clustering/` — the estimators (`KMeans`, `SpectralClustering`, `AgglomerativeClustering`, `SOM`), shared types, and the `Clustering` init namespace.
- `src/eigen/` — eigendecomposition routines (QR, Jacobi, Lanczos, post-processing).
- `src/graph/` — affinity, Laplacian, and connected-component construction.
- `src/distance/` — pairwise distance computation.
- `src/model_selection/` — choosing the number of clusters (`find_optimal_clusters`, `compute_wss`, `kneedle`).
- `src/validation/` — clustering metrics (silhouette, Davies–Bouldin, Calinski–Harabasz, ARI, NMI).
- `src/tensor/` — tensor conversion helpers and type guards.
- `src/random/` — deterministic RNG (Mersenne Twister).
- `src/datasets/` — synthetic dataset generators.
- `src/visualization/` — SOM visualization helpers.

Test files are **colocated** with the source they test (`foo.ts` ↔ `foo.test.ts`); there is no separate `test/` tree. Shared reference fixtures live under `__fixtures__/`.
