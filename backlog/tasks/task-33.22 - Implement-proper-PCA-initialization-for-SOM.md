---
id: task-33.22
title: Implement proper PCA initialization for SOM
status: To Do
assignee: []
created_date: '2025-09-03 06:25'
labels: []
dependencies: []
parent_task_id: task-33
---

## Description

Current PCA initialization is simplified and doesn't compute proper eigenvectors. Need to implement full PCA using eigendecomposition or SVD to initialize SOM weights along principal components of the data.

## Acceptance Criteria

- [ ] Covariance matrix computed correctly
- [ ] Eigendecomposition or SVD implemented
- [ ] Weights initialized along principal components
- [ ] Handles edge cases (fewer samples than dimensions)
- [ ] Performance comparable to sklearn PCA initialization
