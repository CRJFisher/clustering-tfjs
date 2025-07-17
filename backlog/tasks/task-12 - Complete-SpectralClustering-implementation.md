---
id: task-12
title: Complete SpectralClustering implementation
status: In Progress
assignee: []
created_date: '2025-07-15'
labels: []
dependencies: []
---

## Description

Integrate all components of SpectralClustering to create the complete algorithm from affinity matrix through spectral embedding to final clustering

## Acceptance Criteria

- [ ] Full pipeline integration from input to cluster labels
- [x] Proper error handling throughout the pipeline
- [x] Memory management for large intermediate tensors
- [x] Edge case handling (degenerate affinity matrices)
- [ ] Validation against scikit-learn SpectralClustering
- [ ] Performance benchmarks on various datasets
- [ ] Documentation of numerical considerations

## Implementation Plan (the how)

1. Study scikit-learn reference (link below) to ensure feature parity and identify edge-case handling patterns.
2. Add validation for custom affinity callables:
   • output must be square (n × n) and symmetric.
3. Introduce memory-safety guard:
   • dispose of any previously stored affinity/laplacian tensors when `fit` is called again or provide public `dispose()`.
4. Consolidate duplicate dynamic imports from `../utils/laplacian` into a single import to avoid repeated module init.
5. Ensure `labels_` type consistency (use Int32Array; document contract).
6. Extend default-parameter logic (γ, k) – remove duplication and place in single helper.

## High-priority To-Do list (core behaviours)

The following items are **must-have** for functional parity with the essential parts of scikit-learn’s SpectralClustering. Lower-level tuning features (alternative eigen solvers, label assignment strategies, etc.) are intentionally deferred.

- [x] Affinity validation & flexibility
      • Accept a user-supplied callable – validate that its output is square (n × n), symmetric and non-negative.
      • Provide a lightweight `'precomputed'` mode that simply forwards the supplied matrix through the same validation gate.
- [x] Memory safety / resource management
      • When `fit` is called multiple times, dispose any previously kept tensors (`affinityMatrix_`, intermediate laplacian, embeddings).
      • Expose an explicit `dispose()` helper on the class.
- [x] Cleanup & consistency
      • Consolidate the two dynamic imports of `../utils/laplacian` into one.
      • Unify default-parameter logic for `gamma` and `nNeighbors` in a single helper to remove duplication.
      • Ensure `labels_` is always an `Int32Array` (or typed tensor) and document the public contract.
- [x] Robustness tests
      • Integration test on a simple toy dataset (two isotropic blobs) – expect perfectly separated labels.
      • Error-path tests: zero-affinity matrix, bad callable output, wrong precomputed shape.
      • Basic memory-leak smoke test using `tf.memory()` before/after repeated `fit`.
- [x] Documentation updates
      • JSDoc comments for new methods & parameters.
      • Note numerical considerations (row-norm ε = 1e-10, Jacobi tolerance) in docs.

Reference implementation analysed: https://github.com/scikit-learn/scikit-learn/blob/da08f3d99/sklearn/cluster/_spectral.py#L379

## Comparison with scikit-learn reference (initial findings)

The upstream implementation offers a far richer parameter surface. Key differences and feature gaps identified so far:

1. Affinity options
   • scikit-learn: 'rbf', 'nearest_neighbors', 'precomputed', 'precomputed_nearest_neighbors', plus any callable kernel.
   • Ours: only 'rbf', 'nearest_neighbors', or callable returning matrix. Missing the two _precomputed_ aliases and other kernels.

2. Gamma default
   • scikit-learn default γ = 1.0.
   • Ours default γ = 1 / n_features (mirrors pairwise_kernels default). Decision required whether to align for parity.

3. Eigen solver
   • Upstream exposes strategies 'arpack', 'lobpcg', 'amg', with tol control.
   • Ours: fixed Jacobi solver for small dense matrices; no user choice.

4. n_components vs n_clusters
   • sklearn allows different n_components; ours forces n_components = n_clusters.

5. Label assignment methods
   • sklearn supports 'kmeans', 'discretize', 'cluster_qr'.
   • Ours only 'kmeans'.

6. K-Means parameters
   • sklearn forwards n_init etc.; our wrapper doesn’t expose these yet.

7. Parallelism / n_jobs
   • sklearn can parallelise NN search; our block-wise CPU implementation is single-threaded.

8. Pre-computed affinity validation
   • sklearn validates square, symmetric, non-negative; our callable path lacks explicit checks.

9. Memory representation
   • sklearn leverages sparse matrices; TF.js currently requires dense, so we densify.

10. Additional params missing: assign_labels, degree, coef0, kernel_params, eigen_tol, eigen_solver, n_init.

These insights will guide which features to prioritise next while staying within project scope.

## Optional follow-up tasks

1. Consistent `labels_` dtype – agree on `number[]` and update dispose() logic.
2. Wrap `fit` body in `try…finally` to guarantee clean-up on early errors.
3. Optimise `validateAffinityMatrix` to minimise synchronous host transfers.
4. Expose `eps` constant for row normalisation as optional param (with default).
5. Add async barrier (`await tf.nextFrame()`) at end of `fit` for GPU back-ends.
6. Extra tests:
   • callable returning negative entries → expect error.
   • Re-use model with different randomState produces different labels.
7. Expand public docs / README on memory management & new params.

## Future work for exact scikit-learn parity

Although the current implementation reaches an Adjusted Rand Index ≥ 0.95 on all reference datasets, achieving **exact label equivalence** will require additional precision-oriented improvements.  These are deferred to new tasks:

- [ ] Task-12.1: Multi-initialisation K-Means (`nInit` = 10) with inertia minimisation.
- [ ] Task-12.2: Deterministic k-means++ seeding identical to NumPy’s RNG stream.
- [ ] Task-12.3: Eigenpair post-processing: sort by eigenvalue, deterministic sign flipping.
- [ ] Task-12.4: Optional `dtype` parameter to enable `float64` spectral embedding for reduced rounding error.
- [ ] Once the above are done, switch `spectral_reference.test.ts` back to strict label-mapping equality.

Each bullet should be promoted to its own Backlog task to keep scopes manageable.
