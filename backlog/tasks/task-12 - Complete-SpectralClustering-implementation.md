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
- [ ] Proper error handling throughout the pipeline
- [ ] Memory management for large intermediate tensors
- [ ] Edge case handling (degenerate affinity matrices)
- [ ] Validation against scikit-learn SpectralClustering
- [ ] Performance benchmarks on various datasets
- [ ] Documentation of numerical considerations

## High-priority To-Do list (core behaviours)

The following items are **must-have** for functional parity with the essential parts of scikit-learn’s SpectralClustering.  Lower-level tuning features (alternative eigen solvers, label assignment strategies, etc.) are intentionally deferred.

- [x] Affinity validation & flexibility
   • Accept a user-supplied callable – validate that its output is square (n × n), symmetric and non-negative.
   • Provide a lightweight `'precomputed'` mode that simply forwards the supplied matrix through the same validation gate.
- [ ] Memory safety / resource management
   • When `fit` is called multiple times, dispose any previously kept tensors (`affinityMatrix_`, intermediate laplacian, embeddings).
   • Expose an explicit `dispose()` helper on the class.
- [ ] Cleanup & consistency
   • Consolidate the two dynamic imports of `../utils/laplacian` into one.
   • Unify default-parameter logic for `gamma` and `nNeighbors` in a single helper to remove duplication.
   • Ensure `labels_` is always an `Int32Array` (or typed tensor) and document the public contract.
- [ ] Robustness tests
   • Integration test on a simple toy dataset (two isotropic blobs) – expect perfectly separated labels.
   • Error-path tests: zero-affinity matrix, bad callable output, wrong precomputed shape.
   • Basic memory-leak smoke test using `tf.memory()` before/after repeated `fit`.
- [ ] Documentation updates
   • JSDoc comments for new methods & parameters.
   • Note numerical considerations (row-norm ε = 1e-10, Jacobi tolerance) in docs.

Reference implementation analysed: https://github.com/scikit-learn/scikit-learn/blob/da08f3d99/sklearn/cluster/_spectral.py#L379

## Comparison with scikit-learn reference (initial findings)

The upstream implementation offers a far richer parameter surface.  Key differences and feature gaps identified so far:

1. Affinity options
   • scikit-learn: 'rbf', 'nearest_neighbors', 'precomputed', 'precomputed_nearest_neighbors', plus any callable kernel.
   • Ours: only 'rbf', 'nearest_neighbors', or callable returning matrix. Missing the two *precomputed* aliases and other kernels.

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
