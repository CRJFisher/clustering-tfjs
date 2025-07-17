# task-12.3 - Eigenpair post-processing: eigenvalue sorting & deterministic sign flipping

## Description (the why)

When computing the spectral embedding, we rely on an eigendecomposition of the (normalised) graph Laplacian.  Numerical eigensolvers (including TensorFlow.js’ Jacobi and most BLAS/LAPACK routines) exhibit two sources of non-determinism that cause cluster labels to deviate from scikit-learn’s output even when the underlying eigen-subspace is identical:

1. **Unordered eigenpairs** – The solver may return eigenvectors in arbitrary order when their eigenvalues are close, while scikit-learn explicitly sorts them in ascending order of eigenvalue magnitude.
2. **Random sign ambiguity** – Each individual eigenvector is only defined up to a global ± sign.  scikit-learn resolves this ambiguity by flipping signs so that the largest (in absolute value) component of each eigenvector is positive.  This deterministic post-processing ensures that downstream k-means sees the same embedding on every run.

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

*TBD*

