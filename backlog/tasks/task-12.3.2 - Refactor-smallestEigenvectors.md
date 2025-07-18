---
id: task-12.3.2
title: Refactor smallest_eigenvectors to use deterministic post-processing
status: Done
assignee: []
created_date: '2025-07-17'
labels: []
dependencies: [task-12.3.1]
---

## Description (the why)

`smallest_eigenvectors()` still implements its own ordering/sign logic. After introducing `deterministic_eigenpair_processing()` (task-12.3.1) we want a single source of truth. The function also needs to expose _k + 1_ vectors so callers can drop the trivial eigenvector.

## Acceptance Criteria (the what)

- [x] `smallest_eigenvectors(matrix, k)` internally:
  1. Calls Jacobi solver.
  2. Delegates ordering & sign fixing to `deterministic_eigenpair_processing()`.
  3. Returns the **first k + 1 columns** (λ₀,…,λ_k) as `tf.Tensor2D`.
- [x] Implementation passes existing numerical precision tests.
- [x] No duplicated code fragments for sorting / sign flip remain in `laplacian.ts`.
- [x] All tensors created inside the function are properly disposed to avoid leaks.

## Implementation Plan (the how)

1. Remove old inline logic from `laplacian.ts`.
2. Use the helper from task-12.3.1.
3. Update comments & JSDoc to document “k + 1 incl. trivial component” contract.

## Dependencies

- Depends on completion of task-12.3.1.

## Implementation Notes

## Implementation Notes (2025-07-18)

Approach taken

• Re-implemented `smallest_eigenvectors()` to be a thin wrapper around:

1. `jacobi_eigen_decomposition()` – full symmetric eigensolver.
2. `deterministic_eigenpair_processing()` – shared ordering & sign-flip logic introduced in task-12.3.1.

• The helper returns _all_ sorted/sign-fixed eigenvectors. We slice the first _k + 1_ columns (incl. λ₀ constant vector) into a new JS array and convert that into a `tf.Tensor2D` which we return. The complete operation is wrapped in `tf.tidy()` to ensure temporaries are disposed.

• All previous ad-hoc sorting / sign logic was removed from `laplacian.ts`, so a single source of truth is enforced.

• JSDoc updated to document the new contract (returns _k + 1_ vectors).

Verification

• All Laplacian utility and numerical edge-case tests pass.
• Manual inspection & `rg` confirmed no remaining duplicate sort/sign-flip snippets in `laplacian.ts`.
• TensorFlow memory leak check via `tf.memory()` shows no growth across repeated calls.

Limitations / Next Steps

SpectralClustering now consumes `smallest_eigenvectors` correctly but reference-parity tests still fail due to downstream embedding differences. This is expected and will be addressed in task-12.3.3.
