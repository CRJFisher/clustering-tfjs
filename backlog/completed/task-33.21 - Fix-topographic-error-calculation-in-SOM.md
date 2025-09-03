---
id: task-33.21
title: Fix topographic error calculation in SOM
status: Done
updated_date: '2025-09-03 07:45'
assignee: []
created_date: '2025-09-03 06:24'
labels: []
dependencies: []
parent_task_id: task-33
---

## Description

The topographic error calculation is currently broken - it always returns 100% error because isNeighbor is hardcoded to false. Need to properly find second BMU and check if first and second BMUs are neighbors based on grid topology.

## Acceptance Criteria

- [x] findSecondBMU properly integrated
- [x] Neighbor checking implemented for rectangular topology
- [x] Neighbor checking implemented for hexagonal topology
- [x] Tests validate correct topographic error
- [x] Function returns meaningful values between 0 and 1

## Implementation Notes

### Completed Fix
The topographic error calculation was completely broken - it always returned 100% error because `isNeighbor` was hardcoded to `false`. This has been completely rewritten.

### What Was Implemented
1. **Proper BMU finding**: Now correctly calls `findBMU()` and `findSecondBMU()` for each sample
2. **areNeighbors() method**: New private method that checks if two neurons are neighbors
3. **Rectangular topology**: 4-connected neighbors (up, down, left, right)
4. **Hexagonal topology**: 6-connected neighbors with even/odd row offset handling
5. **Memory management**: Proper tensor disposal in the loop

### Key Changes
- Removed placeholder comments and simplified approach
- Actually uses the existing `findSecondBMU` function
- Returns real values between 0 and 1 based on actual neighbor relationships
- Handles both topologies correctly
