---
id: TASK-52.2
title: Align near-zero eigenvalue tolerance between Lanczos and Jacobi paths
status: To Do
assignee: []
created_date: '2026-06-10 08:55'
labels:
  - bug
  - confirmed
dependencies: []
references:
  - 'src/eigen/smallest_eigenvectors_with_values.ts:72'
parent_task_id: TASK-52
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

smallest_eigenvectors_with_values uses tolerance 1e-2 in the Lanczos path and 1e-7 in the Jacobi path when counting near-zero eigenvalues to determine how many extra eigenvectors to return. A matrix with an eigenvalue of ~0.005 gets k+1 eigenvectors from Lanczos but k from Jacobi, so spectral clustering silently produces different embedding dimensions (and therefore wrong cluster labels) for the same matrix depending on which path the n=100 boundary routes it to.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [ ] #1 Both Lanczos and Jacobi paths use the same near-zero eigenvalue tolerance
- [ ] #2 A test verifies that a matrix with a small-but-nonzero eigenvalue produces the same number of returned eigenvectors regardless of which path is taken
- [ ] #3 No change to the expected cluster labels on existing fixtures
- [ ] #4 SpectralClustering produces the same cluster labels on inputs of size 99 and 101 with otherwise identical structure, confirming the n=100 path boundary does not affect results (code-path equivalence test)
<!-- AC:END -->
