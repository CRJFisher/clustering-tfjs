---
id: task-12.18
title: Handle disconnected graph components in spectral clustering
status: To Do
assignee: []
created_date: '2025-07-20'
labels: []
dependencies: []
parent_task_id: task-12
---

## Description

Debug output shows some affinity graphs have multiple disconnected components. Need to implement sklearn's approach for handling these cases robustly.

**Note from cluster analysis**: While overlapping clusters are common in our test data (8/12 fixtures), disconnected components are a separate issue that can occur with sparse k-NN graphs or very small RBF gamma values. This is particularly relevant for the failing circles datasets where concentric rings might create disconnected components with certain affinity parameters.

## Acceptance Criteria

- [ ] Add connected components check to spectral pipeline
- [ ] Implement sklearn's disconnected components handling
- [ ] Test with artificially disconnected data
- [ ] Ensure graceful handling of edge cases

## Implementation Notes

### Prior Findings - OVERLAPS WITH TASK 12.15

From investigations in Tasks 12.10/12.11/12.18:

1. **blobs_n2 has 3 components**: The k-NN graph creates disconnected components
2. **sklearn's handling**: 
   - Issues warning: "Graph is not fully connected, spectral embedding may not work as expected"
   - But continues processing and achieves ARI = 1.0
   - No special handling found in sklearn code
3. **Component detection**: Number of near-zero eigenvalues indicates number of components
4. **Our current handling**: Already returns all zero-eigenvalue eigenvectors

### Overlap with Task 12.15

This task significantly overlaps with Task 12.15 (Component-aware eigenvector selection). Consider:
- Merging these tasks, or
- Making this task focus on component detection/warning
- Task 12.15 focuses on eigenvector selection strategy

### Key Insight

sklearn doesn't seem to have special "handling" for disconnected components beyond:
1. Detecting and warning about it
2. Using eigenvector recovery (which we're missing)
3. Letting k-means handle the component grouping
