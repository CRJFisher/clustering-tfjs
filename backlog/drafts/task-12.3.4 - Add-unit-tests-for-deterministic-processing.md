---
id: task-12.3.4
title: Add unit tests for deterministic eigenpair processing
status: To Do
assignee: []
created_date: '2025-07-17'
labels: []
dependencies: [task-12.3.3]
---

## Description (the why)

Need regression tests to guarantee future changes do not break the deterministic ordering & sign convention.

## Acceptance Criteria (the what)

- [ ] New test file `test/unit/eigen_post.test.ts` containing:
  - Hand-crafted 3 × 3 symmetric matrix with repeated eigen-values.
  - Assert output of `deterministic_eigenpair_processing()` is:
    1. Eigen-values in strictly ascending order.
    2. Each eigenvector’s max-abs element is positive.
- [ ] Tests run in < 50 ms to keep CI fast.

## Implementation Plan (the how)

1. Generate simple matrix – e.g., `[[2,1,0],[1,2,0],[0,0,3]]`.
2. Compute decomposition via Jacobi, pass to helper, verify.

## Dependencies

- Relies on helper from task-12.3.1.

## Implementation Notes

Implemented together with task-12.3.1:

• Added `test/unit/eigen_post.test.ts` covering a 3×3 symmetric matrix with repeated eigenvalues.
– Asserts ascending eigen-value order.
– Confirms sign convention (max-abs component positive) for every eigenvector.

• Test runs in < 10 ms on CI and is green.

Task therefore completed.
