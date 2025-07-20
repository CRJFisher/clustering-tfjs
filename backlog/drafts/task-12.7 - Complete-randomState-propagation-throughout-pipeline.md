---
id: task-12.7
title: Complete randomState propagation throughout pipeline
status: To Do
assignee: []
created_date: '2025-07-19'
labels: []
dependencies: []
parent_task_id: task-12
---

## Description

Some components still use Math.random() instead of the provided RandomState. This includes the Jacobi eigensolver and potentially other utilities. Non-deterministic behavior at any stage prevents achieving exact sklearn parity. Need to thread RandomState through all stochastic operations.

## Acceptance Criteria

- [ ] Pass randomState to Jacobi solver in laplacian utils
- [ ] Remove all Math.random() calls from the pipeline
- [ ] Ensure k-NN tie-breaking uses randomState
- [ ] Add determinism test with fixed seed
