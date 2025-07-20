---
id: task-12.13
title: Investigate affinity matrix sparsity handling
status: To Do
assignee: []
created_date: '2025-07-20'
labels:
  - spectral
  - affinity
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
