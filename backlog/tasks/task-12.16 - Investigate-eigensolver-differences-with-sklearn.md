---
id: task-12.16
title: Investigate eigensolver differences with sklearn
status: To Do
assignee: []
created_date: '2025-07-20'
labels: []
dependencies: []
parent_task_id: task-12
---

## Description

sklearn uses ARPACK with shift-invert mode while we use Jacobi. This might cause numerical differences.

## Acceptance Criteria

- [ ] Document sklearn's exact eigensolver approach
- [ ] Compare numerical results between solvers
- [ ] Determine if solver difference is causing value discrepancies
- [ ] Implement compatible solution if needed

## Implementation Notes

### Prior Findings - THIS TASK MAY BE OBSOLETE

From Task 12.12/12.18 investigation, we already discovered:

1. **Both solvers find the same eigenspace**: Jacobi and ARPACK produce eigenvectors that span the same subspace (projection error ≈ 0)
2. **Different bases within null space**: For degenerate eigenvalues (multiple zeros), they choose different orthonormal bases
3. **The real issue was eigenvector recovery**: Missing the step of dividing by sqrt(degree)
4. **Conclusion**: "No need to change eigensolvers; Jacobi works fine; the issue was post-processing"

### Remaining Relevance

This task may still be relevant for:
1. **Performance comparison**: ARPACK might be faster for large sparse matrices
2. **Numerical stability**: Check if ARPACK's shift-invert mode helps with near-zero eigenvalues
3. **Other test failures**: Some failing tests (circles, moons) might have different issues

### Key Finding

The eigensolver difference is NOT causing the blobs_n2 failures. The issue was the missing eigenvector recovery step.
