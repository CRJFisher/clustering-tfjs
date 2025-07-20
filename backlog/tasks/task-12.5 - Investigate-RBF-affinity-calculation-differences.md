---
id: task-12.5
title: Investigate RBF affinity calculation differences
status: In Progress
assignee:
  - '@chuck'
created_date: '2025-07-20'
updated_date: '2025-07-20'
labels: []
dependencies: []
parent_task_id: task-12
---

## Description

The blobs_n2_rbf test has extremely low ARI (0.064) suggesting fundamental differences in RBF kernel computation. Need to compare our implementation with sklearn's RBF kernel, including gamma parameter interpretation and any data preprocessing.

## Acceptance Criteria

- [ ] Compare RBF kernel values with sklearn for test data
- [ ] Identify any scaling or preprocessing differences
- [ ] Fix RBF implementation to match sklearn
- [ ] blobs_n2_rbf test achieves ARI >= 0.95

## Implementation Plan

1. Create sklearn_reference folder for sklearn code
2. Download and analyze sklearn's RBF kernel implementation
3. Create test script comparing our RBF with sklearn's on blobs_n2_rbf data
4. Identify differences in gamma interpretation or computation
5. Fix our RBF implementation to match sklearn
6. Verify fix improves test results
