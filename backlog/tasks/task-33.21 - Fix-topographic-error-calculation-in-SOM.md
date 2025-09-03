---
id: task-33.21
title: Fix topographic error calculation in SOM
status: To Do
assignee: []
created_date: '2025-09-03 06:24'
labels: []
dependencies: []
parent_task_id: task-33
---

## Description

The topographic error calculation is currently broken - it always returns 100% error because isNeighbor is hardcoded to false. Need to properly find second BMU and check if first and second BMUs are neighbors based on grid topology.

## Acceptance Criteria

- [ ] findSecondBMU properly integrated
- [ ] Neighbor checking implemented for rectangular topology
- [ ] Neighbor checking implemented for hexagonal topology
- [ ] Tests validate correct topographic error
- [ ] Function returns meaningful values between 0 and 1
