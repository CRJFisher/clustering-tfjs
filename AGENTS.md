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

## Comments

Comments describe **WHY**, never **WHAT**. Remove any comment that restates what a name, type, or signature already says — including `@param`/`@returns` for obvious params, field docstrings, section dividers, and step labels. Keep only non-obvious algorithm choices, parity notes (sklearn/scipy/NumPy), sentinel semantics, and mathematical formulas. Test names state what the code does (present tense, no "should" prefix).

## Testing practices

- **Test at scale.** Any method operating on per-sample arrays must have at least one test at n ≥ 300k. JS argument-spread (`Math.max(...array)`) silently crashes above ~200k.
- **Test equivalent paths equivalently.** When the same logic has two implementations behind a size or mode gate, run both on the same input and assert identical results. The boundary value is the highest-risk case.
- **Test state invariants under failure.** For every stateful estimator, verify that a failed `fit()` leaves all output fields in their prior state — not null, not partially updated.
- **Use geometry-specific fixtures, not just blobs.** Generic blob data does not exercise metric edge cases. Cosine tests need antipodal unit vectors; manhattan tests need axis-aligned clusters. Each metric variant needs at least one fixture that would expose a Euclidean-centric implementation.
- **Degenerate inputs are first-class.** All-noise labels, empty input, and single-cluster input must have explicit test cases. Silent return values (0, `[]`) for degenerate inputs are bugs, not acceptable defaults.
