---
id: task-12.24
title: Migrate eigensolver to ml-matrix for 10x performance improvement
status: To Do
assignee: []
created_date: '2025-07-21'
labels: []
dependencies: []
parent_task_id: task-12
---

## Description

Replace our Jacobi eigensolver with ml-matrix's EigenvalueDecomposition which is 10x faster (34ms vs 347ms) and produces identical results. Also see if there are any other performance improvements that can be made by using ml-matrix in other places.

## Acceptance Criteria

- [ ] Replace Jacobi with ml-matrix in smallest_eigenvectors.ts
- [ ] Maintain same API and tensor handling
- [ ] Verify all tests still pass at same rate
- [ ] Performance improvement confirmed
