---
id: TASK-19
title: Sparse kNN affinity and sparse Lanczos for SpectralClustering
status: In Progress
assignee: []
created_date: '2025-07-15'
updated_date: '2026-06-08 15:35'
labels:
  - performance
  - spectral
dependencies:
  - task-19.1
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
SpectralClustering does not scale. Every fit materialises a dense `n × n`
affinity matrix, derives a dense normalised Laplacian from it, and feeds that
dense matrix to the eigensolver. Memory is `O(n²)` and time is dominated by
operations over the full matrix, so the algorithm hits a wall on moderate data:
at `n = 10_000` a fit takes ~140 s (vs ~5 s for the other algorithms) and the
dense matrix alone is ~400 MB.

The `nearest_neighbors` affinity already exists but is stored densely
(`src/graph/affinity.ts`), so it provides no memory benefit. scikit-learn's
answer to this exact problem is a **sparse** k-nearest-neighbour graph carried
all the way through the Laplacian and eigensolver. Our Lanczos solver is already
matrix-free in shape — its core loop calls a `matvec(A, v)` abstraction
(`src/eigen/lanczos.ts`) — but it densifies the input via `arraySync()` before
use. Making the kNN path sparse end-to-end removes the `O(n²)` wall and lets
SpectralClustering scale to large `n`, while preserving exact scikit-learn
output parity.

This work must be validated against scikit-learn for **both correctness and
performance**: cluster assignments must match sklearn
(`affinity='nearest_neighbors'`) within the established tolerance, and the sparse
path must demonstrate the expected memory/time reduction relative to both our
own dense path and sklearn's timings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `SpectralClustering` with `affinity='nearest_neighbors'` runs without
      ever allocating a dense `n × n` matrix (peak memory scales ~`O(n·k)`, not
      `O(n²)`).
- [ ] #2 kNN affinity is produced and stored in a sparse representation (e.g. CSR),
      symmetrised consistently with scikit-learn.
- [ ] #3 The normalised Laplacian is applied as a sparse / matrix-free operator
      (no dense Laplacian materialised) for the sparse path.
- [ ] #4 The Lanczos eigensolver accepts a sparse matrix or `matvec` operator and
      returns the `k` smallest eigenpairs without densifying the operand.
- [ ] #5 Cluster labels for `affinity='nearest_neighbors'` match scikit-learn on
      the existing reference fixtures within the established ARI / label tolerance.
- [ ] #6 A large-`n` case that is currently infeasible on the dense path (e.g.
      `n ≥ 10_000`) completes on the sparse path within available memory.
- [ ] #7 Benchmarks record sparse-path time and peak memory versus (a) the existing
      dense path and (b) scikit-learn's `nearest_neighbors` timings, confirming
      the expected reduction.
- [ ] #8 The dense `rbf` / `precomputed` paths and their existing sklearn parity
      tests remain unchanged and green.
- [ ] #9 Documentation explains when the sparse path activates, its memory/time
      characteristics, and its scikit-learn equivalence.
<!-- AC:END -->



## Implementation Plan (the how)

### Current state (starting point)

- `src/graph/affinity.ts` — `compute_knn_affinity` exists but returns a dense
  `tf.Tensor2D` (line ~54 explicitly notes a sparse representation would be more
  memory efficient).
- `src/graph/laplacian.ts` — `normalised_laplacian(A: tf.Tensor2D)` consumes a
  dense tensor and produces a dense tensor.
- `src/eigen/lanczos.ts` — `lanczos_smallest_eigenpairs(matrix: tf.Tensor2D, …)`
  calls `matrix.arraySync()` once, then iterates via a `matvec(A, v, n)` helper.
  The Lanczos algorithm itself is matrix-free-ready; only the input type and the
  `matvec` implementation assume a dense `number[][]`.
- `src/clustering/spectral.ts` — builds the affinity matrix, warns about
  `O(n²)` memory (~line 168), then runs Laplacian + eigensolve on dense tensors.
- `src/eigen/smallest_eigenvectors_with_values.ts` — already routes `n > 100,
  k < n/3` to Lanczos; this is the path to extend with a sparse operand.

### Phase 1 — Sparse affinity representation

1. Introduce a minimal sparse symmetric matrix type (CSR-style: `indptr`,
   `indices`, `data`) under `src/graph/` with a colocated test.
2. Add a sparse variant of the kNN affinity builder that emits this type
   directly (no dense intermediate). Match scikit-learn's symmetrisation rule
   (`A = max(A, Aᵀ)` / connectivity-vs-distance handling) so graph topology is
   identical.
3. Verify the sparse affinity reproduces the dense `compute_knn_affinity` values
   exactly for small `n` (densify-and-compare in tests only).

### Phase 2 — Matrix-free normalised Laplacian

1. Provide a `matvec` operator for the normalised Laplacian
   `L = I − D^{-1/2} A D^{-1/2}` that consumes the sparse affinity and never
   forms a dense matrix (fold `D^{-1/2}` into the product).
2. Compute the degree vector from the sparse structure.
3. Test the operator against the existing dense `normalised_laplacian` output for
   small `n`.

### Phase 3 — Sparse Lanczos

1. Generalise `lanczos_smallest_eigenpairs` to accept either a dense matrix
   (existing behaviour) or a `matvec` operator + dimension `n`, removing the
   `arraySync()` densification on the sparse path.
2. Implement the sparse `matvec` over the CSR structure.
3. Reuse the existing tridiagonalisation / QL / convergence machinery unchanged.
4. Extend `smallest_eigenvectors_with_values` to thread the sparse operator
   through for the `nearest_neighbors` path.

### Phase 4 — Wire SpectralClustering

1. When `affinity='nearest_neighbors'`, take the sparse path end-to-end and skip
   the dense affinity allocation and the `O(n²)` memory warning.
2. Keep `rbf` and `precomputed` on the existing dense path untouched.
3. Ensure `capture_debug_info` / intermediate-step accessors degrade gracefully
   when no dense matrix exists (expose sparse stats instead).

### Phase 5 — Validate against scikit-learn (correctness + performance)

1. **Correctness:** run the existing `nearest_neighbors` reference fixtures
   through the sparse path; assert label/ARI parity with scikit-learn within the
   established tolerance. Add a fixture at larger `n` if current ones are too
   small to exercise the sparse path meaningfully.
2. **Performance:** extend the benchmark harness to record sparse-path execution
   time and peak memory, compared against the dense path and against
   scikit-learn's `affinity='nearest_neighbors'` timings (sklearn timing harness
   is tracked separately — see the deferred sklearn-timing task). Confirm the
   `O(n²) → O(n·k)` memory reduction and a feasible large-`n` run.

### Phase 6 — Documentation & finalize

1. Document the sparse path in the SpectralClustering docs: activation
   condition, memory/time characteristics, scikit-learn equivalence.
2. `npm run lint` + full test suite green.
3. Tick the acceptance criteria, add Implementation Notes, set status Done.

### Files (anticipated)

- **New:** sparse matrix type + sparse kNN builder under `src/graph/`
  (+ colocated tests), sparse Laplacian operator, sparse `matvec`.
- **Edit:** `src/graph/affinity.ts`, `src/graph/laplacian.ts`,
  `src/eigen/lanczos.ts`, `src/eigen/smallest_eigenvectors_with_values.ts`,
  `src/clustering/spectral.ts`, SpectralClustering docs, benchmark harness.

### Risk / notes

- TensorFlow.js has no native sparse matrix support, so the sparse path lives in
  plain JS/typed arrays — consistent with the existing JS-side Lanczos.
- Symmetrisation and self-loop handling must mirror scikit-learn precisely;
  small topology differences change eigenvectors and break label parity. This is
  the highest-risk correctness area and is gated by the fixture parity tests.
