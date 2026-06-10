---
id: TASK-52.9
title: >-
  Investigate agglomerative compute_distance_matrix missing symmetrisation vs
  pairwise_distance_matrix
status: To Do
assignee: []
created_date: '2026-06-10 08:56'
labels:
  - bug
  - plausible
dependencies: []
references:
  - 'src/clustering/agglomerative.ts:317'
  - src/distance/pairwise_distance.ts
parent_task_id: TASK-52
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

AgglomerativeClustering.compute_distance_matrix re-implements euclidean, manhattan, and cosine pairwise distances inline rather than delegating to pairwise_distance_matrix. The shared implementation in distance/pairwise_distance.ts applies clamping, (D+Dᵀ)/2 symmetrisation, and diagonal zeroing to guarantee a valid distance matrix. The inline version omits these steps; numerical asymmetry can break the NN-chain algorithm's tie resolution and produce different dendrograms than sklearn.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [ ] #1 Investigate whether compute_distance_matrix produces asymmetric matrices in practice on the existing fixture data
- [ ] #2 Either delegate to pairwise_distance_matrix (applying the same symmetrisation), or document why the inline version is intentionally different
- [ ] #3 A test asserts that the distance matrix produced by compute_distance_matrix satisfies |D − Dᵀ|\_max < 1e-10 and diag(D) = 0 for all supported metrics on the existing fixture data (matrix algebraic properties test — unconditional, not gated on whether asymmetry is currently observable)
<!-- AC:END -->
