---
id: task-12
title: Complete SpectralClustering implementation
status: In Progress
assignee: []
created_date: '2025-07-15'
updated_date: '2025-07-21'
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

## Implementation Notes

## Implementation Notes (Updated 2025-07-19 - Part 2)

### Critical findings on sklearn parity issues:

1. **Row normalization investigation**: Initially suspected that sklearn applies row normalization before k-means, but investigation showed that sklearn only applies row normalization when using assign_labels='discretize', NOT for the default k-means approach. ✅ Fixed

2. **Spectral embedding normalization**: Found that sklearn's spectral_embedding function applies a critical normalization step: where dd is the square root of the degree vector. This recovers u = D^{-1/2} x from the eigenvector output x when using the normalized Laplacian. ✅ Implemented

3. **Constant eigenvector handling**: **CRITICAL FINDING** - sklearn KEEPS constant eigenvectors and runs k-means on them\! Our implementation was removing all constant eigenvectors, which is incorrect. ✅ Fixed - now keeping first nClusters eigenvectors

4. **Normalized Laplacian computation**: **CRITICAL FINDING** - Our normalized Laplacian implementation was incorrect. scipy/sklearn:
   - Zeros out diagonal of affinity matrix before computing degrees
   - Ensures diagonal of normalized Laplacian is exactly 1
   - For isolated nodes (degree=0), uses 1 instead of 0 for inverse sqrt
     ✅ Fixed our implementation to match this behavior

5. **Eigenvalue solver convergence**: **CRITICAL FINDING** - Our Jacobi solver is NOT converging properly:
   - With 100 iterations: off-diagonal norm = 0.378, smallest eigenvalue = 0.039
   - With 1000 iterations: off-diagonal norm = 0.005, smallest eigenvalue = 8.7e-7
   - sklearn/scipy gets eigenvalue ≈ 4.4e-16
   - The Jacobi solver needs significantly more iterations but is very slow

6. **Current status after all fixes**:
   - ✅ Fixed normalized Laplacian computation
   - ✅ Removed constant eigenvector filtering
   - ✅ Added D^{-1/2} normalization
   - ❌ Jacobi solver still not converging to required precision
   - ❌ Tests still failing due to inaccurate eigendecomposition

7. **Next steps**:
   - Consider alternative eigensolver (e.g., power iteration for smallest eigenvalues)
   - Or increase Jacobi tolerance/iterations further
   - Or use TensorFlow.js linalg operations if available
   - Profile to see if we can make Jacobi faster for more iterations

Current Status (2025-07-21):\n- 7/12 tests passing (58%)\n- All k-NN tests pass (disconnected graphs handled correctly)\n- 5 RBF tests fail due to eigenvector numerical accuracy differences\n- Root cause: Jacobi eigensolver produces slightly different results than sklearn's ARPACK\n- Failing tests have ARI ~0.87-0.93 (close but below 0.95 threshold)\n\nNext steps:\n1. Try making k-means more robust (task 12.22)\n2. Investigate alternative eigensolvers (task 12.23)\n3. If above fail, consider ARPACK bindings (task 12.21)
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
   • scikit-learn: 'rbf', 'nearest*neighbors', 'precomputed', 'precomputed_nearest_neighbors', plus any callable kernel.
   • Ours: only 'rbf', 'nearest_neighbors', or callable returning matrix. Missing the two \_precomputed* aliases and other kernels.

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

## Subtasks for sklearn parity (REFOCUSED)

Based on analysis of test failures, particularly k-NN affinity (ARI 0.08-0.63), the following prioritized subtasks address the root causes:

**Phase 1: k-NN Affinity Fixes (CRITICAL)**

- [ ] Task-12.1: Fix k-NN default nNeighbors to match sklearn: `round(log2(n_samples))`
- [ ] Task-12.2: Add k-NN graph connectivity check with self-loops for disconnected graphs

**Phase 2: Core Algorithm Fixes**

- [ ] Task-12.3: Drop ALL trivial eigenvectors in spectral embedding (not just first)
- [ ] Task-12.4: Fix row normalization to use `max(norm, eps)` instead of `norm + eps`
- [ ] Task-12.5: Throw error instead of zero-padding when insufficient eigenvectors

**Phase 3: Additional Parity Fixes**

- [ ] Task-12.6: Align k-means empty cluster handling with sklearn
- [ ] Task-12.7: Complete randomState propagation throughout pipeline (Jacobi solver, etc.)

Once all subtasks are complete, all fixtures should pass with ARI ≥ 0.95.

## Specific Test Fixtures We're Trying to Fix

As of task 12.13, we have 12 fixture tests in `test/fixtures/spectral/`:

**Currently Passing (4/12):**
- `blobs_n2_knn.json` - ARI = 1.0 ✓
- `blobs_n2_rbf.json` - ARI = 1.0 ✓
- `blobs_n3_knn.json` - ARI = 1.0 ✓
- `blobs_n3_rbf.json` - ARI = 1.0 ✓

**Currently Failing (8/12):**
- `circles_n2_knn.json` - ARI = 0.3094 ✗
- `circles_n2_rbf.json` - ARI = 0.9333 ✗
- `circles_n3_knn.json` - ARI = 0.8992 ✗
- `circles_n3_rbf.json` - ARI = 0.7675 ✗
- `moons_n2_knn.json` - ARI = 0.6339 ✗
- `moons_n2_rbf.json` - ARI = 0.9333 ✗
- `moons_n3_knn.json` - ARI = 0.9453 ✗
- `moons_n3_rbf.json` - ARI = 0.4144 ✗

The goal is to achieve ARI ≥ 0.95 for all 12 fixtures to match sklearn's performance.

## Implementation Notes (Updated)

Many determinism and algorithm fixes have been completed:

- ✅ Multi-init k-means with inertia minimization (nInit=10 default)
- ✅ Deterministic k-means++ seeding aligned with NumPy
- ✅ Deterministic KNN tie-breaking
- ✅ Basic eigenpair processing with sign flipping
- ✅ RandomState propagation to k-means
- ✅ RBF gamma default aligned to 1/n_features

The remaining subtasks focus on the critical differences preventing k-NN parity.
