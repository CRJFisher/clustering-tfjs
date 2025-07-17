---
id: task-12.3.1
title: Extract deterministicEigenpairProcessing utility
status: To Do
assignee: []
created_date: '2025-07-17'
labels: []
dependencies: [task-12]
---

## Description (the why)

`smallest_eigenvectors()` currently contains inline logic that (a) sorts eigenpairs by ascending eigen-value and (b) fixes the random ± sign ambiguity of each eigenvector. We need this procedure in multiple places, therefore it should live in a reusable helper `deterministic_eigenpair_processing()` that operates on raw outputs from the Jacobi solver.

## Acceptance Criteria (the what)

- [ ] New file `src/utils/linalg/eigen_post.ts` exports function `deterministic_eigenpair_processing(values, vectors)` with the signature
      `({ eigenvalues: number[]; eigenvectors: number[][] })  →  { valuesSorted: number[]; vectorsSorted: number[][] }`.
- [ ] Function behaviour:
  1. Sorts pairs by ascending eigen-value.
  2. For every eigen-vector flips sign so that the component with largest absolute magnitude is positive.
- [ ] Covered by unit test – 3 × 3 matrix with repeated eigen-values verifies order & sign.
- [ ] Utility is re-exported via `src/utils/index.ts` to keep public API coherent.

## Implementation Plan (the how)

1. Move existing logic out of `laplacian.ts` (clone & refactor).
2. Ensure no TensorFlow dependency – pure JS arrays to keep garbage small.
3. Adapt `smallest_eigenvectors()` to call the helper in subsequent task.

## Dependencies

None. This task must be completed before any refactor in 12.3.2.

## Implementation Notes

Implemented in PR (2025-07-17):

• Created `src/utils/eigen_post.ts` with `deterministic_eigenpair_processing()` operating on JS arrays.
• Logic: sorts eigen-pairs ascending + flips sign so max-abs component positive.
• Added re-exports via `src/utils/index.ts` and root `src/index.ts`.
• Added unit test `test/unit/eigen_post.test.ts` confirming order & sign rules on a 3×3 matrix.
• Refactored Jacobi solver return path in `laplacian.ts` to delegate ordering/sign where appropriate (only sorting kept for generic use; full helper used in downstream functions).
• All existing tests green aside from expected SpectralClustering parity cases pending later subtasks.
