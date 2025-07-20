---
id: task-12.16
title: Investigate affinity matrix sparsity handling
status: To Do
assignee: []
created_date: '2025-07-20'
updated_date: '2025-07-20'
labels:
  - spectral
  - affinity
  - optimization
  - deprioritized
dependencies: []
parent_task_id: task-12
---

## Description

sklearn can leverage sparse matrices for efficiency, while our implementation uses dense matrices. Investigate if this difference in matrix representation is causing numerical differences, especially for k-NN affinity where the matrix is naturally sparse.

## Acceptance Criteria

- [ ] Document sklearn's sparse matrix handling approach
- [ ] Identify any numerical differences from dense representation
- [ ] Implement workarounds if sparse handling affects results
- [ ] k-NN affinity tests show improved results

## Implementation Notes

Current dense matrix implementation is functionally correct. Sparse matrix support is an optimization that can be addressed after achieving functional parity with sklearn. Deprioritizing to focus on correctness first.
