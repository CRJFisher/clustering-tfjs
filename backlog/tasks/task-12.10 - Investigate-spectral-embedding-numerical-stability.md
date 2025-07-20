---
id: task-12.10
title: Investigate spectral embedding numerical stability
status: To Do
assignee: []
created_date: '2025-07-20'
labels:
  - spectral
  - numerical-stability
dependencies: []
parent_task_id: task-12
---

## Description

The very low ARI scores (0.088) for blobs datasets suggest fundamental numerical issues. Investigate numerical stability in the spectral embedding pipeline, including: conditioning of the Laplacian matrix, numerical errors in eigendecomposition, and potential issues with near-zero eigenvalues.

## Acceptance Criteria

- [ ] Laplacian matrix conditioning is analyzed and documented
- [ ] Near-zero eigenvalue handling matches sklearn
- [ ] Numerical stability improvements are implemented
- [ ] Blobs dataset achieves ARI > 0.9
