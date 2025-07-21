---
id: task-12.19
title: Improve spectral clustering for overlapping clusters
status: To Do
assignee: []
created_date: '2025-07-20'
updated_date: '2025-07-21'
labels: []
dependencies: []
parent_task_id: task-12
---

## Description

Analysis shows 8/12 fixtures have overlapping clusters, particularly circles (concentric rings) and moons (interleaved crescents). While spectral clustering is designed for non-convex shapes, our implementation struggles with these cases. k-NN affinity performs better than RBF on overlapping data, suggesting affinity construction is key.

## Acceptance Criteria

- [ ] Document sklearn's approach to overlapping clusters
- [ ] Analyze why k-NN outperforms RBF on overlapping data
- [ ] Investigate affinity matrix properties for overlapping vs separated clusters
- [ ] Improve handling of circles and moons datasets

## Implementation Plan

1. This task may no longer be relevant - the issue is likely numerical accuracy, not overlapping clusters\n2. Investigate if any of the failing tests actually have overlapping clusters\n3. If not, consider closing this task or refocusing on numerical accuracy\n4. The failing tests (circles and moons) are actually well-separated clusters
