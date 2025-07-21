---
id: task-12.22
title: Fix k-NN default nNeighbors to match sklearn
status: To Do
assignee: []
created_date: '2025-07-21'
labels: []
dependencies: []
parent_task_id: task-12
---

## Description

Our k-NN implementation uses a fixed default of 10 neighbors, while sklearn uses round(log2(n_samples)). This could affect graph connectivity and clustering results, especially for the failing circles_n3_knn test.

## Acceptance Criteria

- [ ] Default nNeighbors matches sklearn formula: round(log2(n_samples))
- [ ] k-NN tests produce correct graph structure
- [ ] No regression in existing tests
