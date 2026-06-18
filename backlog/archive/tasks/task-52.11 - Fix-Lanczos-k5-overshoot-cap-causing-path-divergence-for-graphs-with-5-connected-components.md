---
id: TASK-52.11
title: >-
  Fix Lanczos k+5 overshoot cap causing path divergence for graphs with >5
  connected components
status: Done
assignee: []
created_date: '2026-06-10 10:30'
labels:
  - bug
  - confirmed
dependencies: []
parent_task_id: TASK-52
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

In smallest_eigenvectors_with_values, the Lanczos path requests k_request = min(k+5, n) eigenpairs to allow for near-zero eigenvalue detection. For graphs with more than 5 connected components, the near-zero count c > 5, but slice_cols is capped at k+5. The Jacobi path performs a full decomposition and correctly counts all near-zeros up to n. This means for graphs with >5 disconnected components, the two paths return different numbers of eigenvectors at the n=100 boundary — the same class of divergence that task-52.2 fixed for the tolerance case.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [x] #1 Lanczos and Jacobi paths return the same number of eigenvectors for matrices with >5 structural zeros,A test verifies path equivalence for a matrix with 6 or more structural zeros,No regression in existing test suite
<!-- AC:END -->

## Implementation Notes

### High-level summary

When a graph has more than 5 disconnected components, `smallest_eigenvectors_with_values` must include all c near-zero eigenvalues (one per component) in the spectral embedding. The Lanczos path's `k_request = min(k + 5, n)` buffer was insufficient for two compounding reasons: the fixed "+5" cap could be smaller than c, and Lanczos itself requires a Krylov subspace sized roughly `c + 5` to fully resolve the degenerate zero eigenspace — so for small k relative to c, the initial call may find only some of the zeros.

The fix replaces the single-pass buffer with an expansion loop in `lanczos_path`. Starting from `k + 5`, the loop re-runs Lanczos with `k_cur += c + 5` until the near-zero count `c` stabilizes (no new zeros found by expanding the subspace). When `c` stops growing, Lanczos has fully explored the zero eigenspace and `slice_cols = k + c` matches what the Jacobi path returns from a full decomposition.

The loop terminates because each iteration either increases `c` (finding more zeros) or exhausts the budget at `k_cur = n`. In practice it converges in two or three Lanczos calls: the first finds a lower bound on `c`, the second resolves all zeros, and the third confirms stability. The extracted `count_near_zeros()` helper removes the duplicate counting logic that previously existed in both the Lanczos and Jacobi paths.

Three parameterized test cases verify path equivalence for block-diagonal complete-graph Laplacians with 6 and 8 components at k=2 and k=6, covering both the original "cap" scenario (k ≈ c) and the "under-resolution" scenario (k ≪ c).
