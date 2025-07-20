---
id: task-12.7
title: Align k-means empty cluster handling with sklearn
status: To Do
assignee: []
created_date: '2025-07-19'
labels: []
dependencies: []
parent_task_id: task-12
---

## Description

When k-means encounters empty clusters, sklearn uses a specific reseeding strategy based on points with highest distance to nearest centroid. Our implementation may handle this differently, causing divergent results. This is especially important with the deterministic multi-init approach.

## Acceptance Criteria

- [ ] Implement sklearn's empty cluster reseeding strategy
- [ ] Use points with highest SSE contribution for new centers
- [ ] Ensure deterministic tie-breaking using RandomState
- [ ] Add test for empty cluster scenario
