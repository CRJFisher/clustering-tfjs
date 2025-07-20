---
id: task-12.3
title: Eigenpair post-processing: eigenvalue sorting & deterministic sign flipping
status: To Do
assignee: []
created_date: '2025-07-17'
labels: []
dependencies: [task-12]
---

## Description (the why)

When computing the spectral embedding, we rely on an eigendecomposition of the (normalised) graph Laplacian. Numerical eigensolvers (including TensorFlow.js’ Jacobi and most BLAS/LAPACK routines) exhibit two sources of non-determinism that cause cluster labels to deviate from scikit-learn’s output even when the underlying eigen-subspace is identical:

1. **Unordered eigenpairs** – The solver may return eigenvectors in arbitrary order when their eigenvalues are close, while scikit-learn explicitly sorts them in ascending order of eigenvalue magnitude.
2. **Random sign ambiguity** – Each individual eigenvector is only defined up to a global ± sign. scikit-learn resolves this ambiguity by flipping signs so that the largest (in absolute value) component of each eigenvector is positive. This deterministic post-processing ensures that downstream k-means sees the same embedding on every run.

Our current implementation forwards the raw eigenvectors to the embedding step without additional processing, leading to arbitrary column permutations and sign flips.

## Acceptance Criteria (the what)

- [ ] Add utility `deterministicEigenpairProcessing(eigenVectors, eigenValues)` that:
  1. Sorts eigenpairs by ascending eigenvalue.
  2. For each eigenvector, finds the entry with the largest absolute value and multiplies the entire vector by −1 if that entry is negative.
- [ ] Integrate this utility in the spectral embedding pipeline after eigendecomposition and before row normalisation.
- [ ] Create unit tests:
  - [ ] Provide handcrafted 3×3 matrix with known degenerate eigenvalues; assert sorted order and sign convention.
  - [ ] Snapshot test with fixture from scikit-learn ensuring identical embedding matrix (up to machine precision) for the first `n_clusters` components.
- [ ] Update docs / JSDoc: document deterministic ordering & sign convention.

## Implementation Plan (the how)

1. Implement helper in `src/utils/linalg/eigen_post.ts` operating on `tf.Tensor2D` and `tf.Tensor1D`.
2. Re-export via `utils/index.ts` to keep public surface coherent.
3. Refactor `SpectralEmbedding.compute()` (or equivalent) to invoke the helper.
4. Ensure any cached tensors are disposed of to prevent memory leaks.
5. Add tests under `test/unit/eigen_post.test.ts`.

## Dependencies

Depends on completed eigendecomposition from task-10 & task-10.1. Complements tasks 12.1 & 12.2 for full label parity.

## Implementation Notes (to fill after completion)

### Findings to date (2025-07-17)

1. Implemented deterministic eigenpair handling in `src/utils/laplacian.ts`.
   - Eigenvectors are sorted by ascending eigen-value.
   - A global ± sign is applied so the entry with the largest absolute value is positive (sklearn rule).

2. Basic Laplacian / Jacobi unit tests stay green → helper appears numerically correct on toy inputs.

3. However the full reference-parity test-suite still fails almost completely (low/NaN ARI). One robustness test now also fails (all samples assigned to the same cluster).

4. Experiment: dropping the first trivial eigenvector (λ≈0) before k-means improves some fixtures → indicates the constant vector is currently included when it should be excluded.

5. Hypothesis: ordering / sign logic is executed _after_ truncation, so we may still end up with permuted columns. Also `smallest_eigenvectors()` always keeps the trivial vector.

### Progress update (2025-07-17 – evening)

✔ 12.3.1 – Extracted `deterministic_eigenpair_processing()` into `src/utils/eigen_post.ts` with unit tests.

✔ 12.3.2 – Refactored `smallest_eigenvectors()` to delegate to the new helper and return _k + 1_ vectors.

❗ Spectral reference-parity tests still fail because the constant eigenvector is still fed into k-means. Addressed next in task 12.3.3.

### Debugging / completion plan

1. Extract the logic into a standalone helper `deterministic_eigenpair_processing()` as originally described by the AC; call it directly from `smallest_eigenvectors()`.
2. `smallest_eigenvectors(matrix, k)` should:
   1. run Jacobi → values & vectors
   2. call the new helper → ordered & sign-fixed pairs
   3. **return k+1 vectors**, then the caller (SpectralClustering) drops the first (trivial) one.
3. Add mandatory unit test with handcrafted 3×3 matrix to assert column order & sign.
4. Adjust `SpectralClustering.fit()` pipeline: request `nClusters+1` vectors, slice off first column, proceed.
5. Re-run full test-suite; if individual fixtures still fail, dump intermediate tensors for comparison with sklearn outputs embedded in JSON fixtures.
6. Verify `randomState` is correctly forwarded to k-means initialisation (otherwise ARI may vary).
7. Final cleanup: memory disposal audit, update JSDoc & README snippets.
