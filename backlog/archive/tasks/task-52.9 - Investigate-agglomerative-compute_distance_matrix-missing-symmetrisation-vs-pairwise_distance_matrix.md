---
id: TASK-52.9
title: >-
  Investigate agglomerative compute_distance_matrix missing symmetrisation vs
  pairwise_distance_matrix
status: Done
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
- [x] #1 Investigate whether compute_distance_matrix produces asymmetric matrices in practice on the existing fixture data
- [x] #2 Either delegate to pairwise_distance_matrix (applying the same symmetrisation), or document why the inline version is intentionally different
- [x] #3 A test asserts that the distance matrix produced by compute_distance_matrix satisfies |D − Dᵀ|\_max < 1e-10 and diag(D) = 0 for all supported metrics on the existing fixture data (matrix algebraic properties test — unconditional, not gated on whether asymmetry is currently observable)
<!-- AC:END -->

## Implementation Notes

## High-level summary

A code review flagged that `compute_distance_matrix` re-implements three pairwise distance metrics inline while the shared `pairwise_distance_matrix` adds clamping, `(D+Dᵀ)/2` symmetrisation, and diagonal zeroing that the inline version appeared to omit.

Investigation shows no bug. The inline implementation is symmetric by construction: for every `(i, j)` pair the loop assigns the same scalar `dist` to both `D[i*n+j]` and `D[j*n+i]`, so `D[i,j] ≡ D[j,i]` exactly, not approximately. The diagonal is zero because `Float64Array` zero-initialises and the loop iterates `j > i`, never touching `D[i*n+i]`. Delegating to `pairwise_distance_matrix` would be incorrect: that function uses TF float32 tensors and Gram-matrix shortcuts whose numerical hazards necessitate clamping and explicit symmetrisation; the NN-chain algorithm requires float64 to reproduce sklearn's merge-order tie resolution exactly.

The JSDoc on `compute_distance_matrix` now carries three explicit guarantees — symmetry by construction, zero diagonal by construction, float64-only rationale — closing the documented ambiguity. A new test describe block `"compute_distance_matrix algebraic properties"` in `agglomerative.test.ts` asserts `|D − Dᵀ|_max < 1e-10` and `diag(D) = 0` for euclidean, manhattan, and cosine on the existing fixture data, making the contract unconditionally machine-verifiable.

The test accesses the private static via TypeScript bracket notation (bracket access bypasses TypeScript's private check without requiring `as any`, which the project bans). The manhattan case reuses the euclidean fixture's coordinates — algebraic properties depend only on the point coordinates, not the metric labels. `precomputed` is excluded because it bypasses `compute_distance_matrix` entirely; the caller supplies the matrix directly.
