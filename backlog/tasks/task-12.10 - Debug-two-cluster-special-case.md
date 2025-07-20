---
id: task-12.10
title: Investigate spectral embedding numerical stability
status: To Do
assignee: []
created_date: '2025-07-20'
updated_date: '2025-07-20'
labels:
  - spectral
  - numerical-stability
dependencies: []
parent_task_id: task-12
---

## Description

The blobs_n2 datasets have catastrophically low ARI scores (0.088) while blobs_n3 passes perfectly. This suggests a fundamental issue with how we handle the two-cluster special case in spectral clustering. Need to debug the n=2 clustering pipeline and compare with sklearn's handling.

**Critical insight from cluster analysis**: The blobs_n2 datasets have EXCELLENT cluster separation (silhouette score 0.741, separation ratio 3.16x, no overlapping clusters). This definitively proves the failure is NOT due to data quality but rather an algorithmic implementation issue specific to 2-cluster cases.
## Acceptance Criteria

- [ ] Create minimal two-cluster reproduction case
- [ ] Compare eigenvector selection for n=2 vs n=3
- [ ] Verify sklearn's two-cluster handling
- [ ] Fix implementation to achieve ARI > 0.95 on blobs_n2
- [ ] Verify fix doesn't break passing tests (especially circles_n2_knn which passes)
