---
id: task-12.12
title: Investigate eigensolver differences with sklearn
status: To Do
assignee: []
created_date: '2025-07-20'
labels: []
dependencies: []
parent_task_id: task-12
---

## Description

sklearn uses ARPACK with shift-invert mode while we use Jacobi. This might cause numerical differences.

## Acceptance Criteria

- [ ] Document sklearn's exact eigensolver approach
- [ ] Compare numerical results between solvers
- [ ] Determine if solver difference is causing value discrepancies
- [ ] Implement compatible solution if needed
