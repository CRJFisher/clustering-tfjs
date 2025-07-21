---
id: task-12.23
title: Investigate alternative eigensolvers for better accuracy
status: To Do
assignee: []
created_date: '2025-07-21'
labels: []
dependencies: []
parent_task_id: task-12
---

## Description

Our Jacobi solver has hit its accuracy limit. Investigate alternative eigendecomposition methods that might provide better accuracy than Jacobi but are easier to implement than full ARPACK bindings.

## Acceptance Criteria

- [ ] Find eigensolver with better accuracy than current Jacobi
- [ ] Implementation feasible in pure TypeScript/JavaScript
- [ ] Improved accuracy on failing RBF tests
