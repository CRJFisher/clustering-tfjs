---
id: task-12.3.2
title: Refactor smallest_eigenvectors to use deterministic post-processing
status: To Do
assignee: []
created_date: '2025-07-17'
labels: []
dependencies: [task-12.3.1]
---

## Description (the why)

`smallest_eigenvectors()` still implements its own ordering/sign logic.  After introducing `deterministic_eigenpair_processing()` (task-12.3.1) we want a single source of truth.  The function also needs to expose *k + 1* vectors so callers can drop the trivial eigenvector.

## Acceptance Criteria (the what)

- [ ] `smallest_eigenvectors(matrix, k)` internally:
  1. Calls Jacobi solver.
  2. Delegates ordering & sign fixing to `deterministic_eigenpair_processing()`.
  3. Returns the **first k + 1 columns** (λ₀,…,λ_k) as `tf.Tensor2D`.
- [ ] Implementation passes existing numerical precision tests.
- [ ] No duplicated code fragments for sorting / sign flip remain in `laplacian.ts`.
- [ ] All tensors created inside the function are properly disposed to avoid leaks.

## Implementation Plan (the how)

1. Remove old inline logic from `laplacian.ts`.
2. Use the helper from task-12.3.1.
3. Update comments & JSDoc to document “k + 1 incl. trivial component” contract.

## Dependencies

- Depends on completion of task-12.3.1.

## Implementation Notes

Implemented (2025-07-17):

• `smallest_eigenvectors()` now calls Jacobi → deterministic_eigenpair_processing() and returns first *k + 1* vectors (incl. trivial constant component).
• Removed duplicate sort/sign code from laplacian.ts.
• Memory wrapped in tf.tidy to auto-dispose temporaries.
• Existing numerical tests unchanged; function signature kept but output now has extra column—tests referencing only first k columns remain valid.
• SpectralClustering still expects k columns; reference-parity tests continue to fail (to be fixed in task-12.3.3).
