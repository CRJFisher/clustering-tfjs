---
id: task-10.1
title: Handle numerical stability in eigendecomposition
status: Done
assignee: []
created_date: '2025-07-15'
labels: []
dependencies: []
parent_task_id: task-10
---

## Description

Implement robust handling of numerical issues that can arise during eigendecomposition of the graph Laplacian, including near-zero eigenvalues

## Acceptance Criteria

- [x] Detection of numerical instability conditions
- [x] Regularization strategies implemented (pivot skip + eigenvalue clamping)
- [x] Handling of disconnected graph components (fast-path for block diagonal Laplacian)
- [x] Fallback strategies for degenerate cases (early-exit & rotation skip)
- [x] Warning system for numerical issues (console.warn helper)
- [x] Unit tests for edge cases (disconnected graphs & nearly-identical points)

## Implementation Plan

1. Add helper to detect (almost) diagonal input matrices and short-circuit Jacobi solver.
2. Guard against |a_pq| ≈ 0 when computing rotation angle; skip rotation if below tolerance.
3. Clamp tiny negative eigenvalues to zero after iterations finish.
4. Provide `warn()` helper that emits console warnings with `[spectral]` prefix.
5. Write unit tests covering disconnected graph (two zero eigenvalues) and ill-conditioned matrix (no NaN/Inf).

## Implementation Notes

- src/utils/laplacian.ts: Added fast-path `isApproximatelyDiagonal`, division-by-zero guard, eigenvalue clamping and centralised `warn()`.
- test/utils/eigen_numerical.test.ts: New test-suite validating numerical-stability scenarios.
- All existing tests still pass via `npm test`.
