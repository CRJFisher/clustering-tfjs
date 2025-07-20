---
id: task-12.15
title: Debug k-means initialization differences
status: To Do
assignee: []
created_date: '2025-07-20'
updated_date: '2025-07-20'
labels:
  - spectral
  - k-means
  - deprioritized
dependencies: []
parent_task_id: task-12
---

## Description

Despite implementing deterministic k-means++ initialization, there may still be subtle differences in how we select initial centers compared to sklearn. Investigate and align the exact k-means++ implementation, including how distances are computed and how random choices are made.

## Acceptance Criteria

- [ ] K-means++ initialization matches sklearn's exact algorithm
- [ ] Random state usage follows sklearn's pattern
- [ ] Initial centers are identical for same random seed
- [ ] Blobs dataset tests show improved clustering

## Implementation Notes

K-means++ initialization is already correctly implemented. Deprioritizing as current implementation matches sklearn's approach. Only revisit if issues persist after fixing two-cluster case and RBF scaling.
