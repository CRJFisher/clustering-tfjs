---
id: TASK-52.11
title: >-
  Fix Lanczos k+5 overshoot cap causing path divergence for graphs with >5
  connected components
status: To Do
assignee: []
created_date: '2026-06-10 10:30'
labels:
  - bug
  - confirmed
dependencies: []
parent_task_id: TASK-52
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

In smallest_eigenvectors_with_values, the Lanczos path requests k_request = min(k+5, n) eigenpairs to allow for near-zero eigenvalue detection. For graphs with more than 5 connected components, the near-zero count c > 5, but slice_cols is capped at k+5. The Jacobi path performs a full decomposition and correctly counts all near-zeros up to n. This means for graphs with >5 disconnected components, the two paths return different numbers of eigenvectors at the n=100 boundary — the same class of divergence that task-52.2 fixed for the tolerance case.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [ ] #1 Lanczos and Jacobi paths return the same number of eigenvectors for matrices with >5 structural zeros,A test verifies path equivalence for a matrix with 6 or more structural zeros,No regression in existing test suite
<!-- AC:END -->
