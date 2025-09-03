---
id: task-33.24
title: Implement proper Mexican hat neighborhood for SOM
status: To Do
assignee: []
created_date: '2025-09-03 06:25'
labels: []
dependencies: []
parent_task_id: task-33
---

## Description

Mexican hat (Ricker wavelet) neighborhood function appears oversimplified. Need proper implementation with lateral inhibition - positive influence near BMU and negative influence at medium distances.

## Acceptance Criteria

- [ ] Correct Ricker wavelet formula implemented
- [ ] Lateral inhibition working (negative values)
- [ ] Parameter tuning for sigma implemented
- [ ] Tests validate shape of influence function
- [ ] Comparison with reference implementation
