---
id: task-12.15
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
