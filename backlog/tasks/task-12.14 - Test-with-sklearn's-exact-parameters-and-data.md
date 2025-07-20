---
id: task-12.14
title: Test with sklearn's exact parameters and data
status: To Do
assignee: []
created_date: '2025-07-20'
labels:
  - spectral
  - testing
dependencies: []
parent_task_id: task-12
---

## Description

Create a test that uses the exact same input data and parameters as sklearn's fixture tests. Export intermediate results from sklearn (affinity matrix, Laplacian, eigenvalues/vectors, embedding) and compare at each step to identify where the divergence occurs.

## Acceptance Criteria

- [ ] Test framework for comparing with sklearn intermediates
- [ ] Intermediate results exported from sklearn
- [ ] Exact divergence point identified
- [ ] Root cause analysis documented
