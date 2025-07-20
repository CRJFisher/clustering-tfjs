---
id: task-12.14
title: Fix RBF gamma scaling for fixture data
status: To Do
assignee: []
created_date: '2025-07-20'
labels: []
dependencies: []
parent_task_id: task-12
---

## Description

Task 12.5 discovered that RBF fixtures use gamma=1.0 but sklearn actually needs much smaller values (0.1-0.7) for the data scales. Either fix the fixtures or implement proper gamma auto-scaling.

**Insights from cluster analysis**:

- RBF affinity generally performs worse than k-NN on overlapping clusters
- All RBF fixtures with overlapping clusters (circles, moons) are failing
- The gamma parameter is critical for handling different cluster densities and overlaps

## Acceptance Criteria

- [ ] Analyze correct gamma values for each RBF fixture
- [ ] Update fixtures OR implement gamma auto-scaling
- [ ] Test RBF performance with proper scaling
- [ ] Achieve comparable results to k-NN tests
