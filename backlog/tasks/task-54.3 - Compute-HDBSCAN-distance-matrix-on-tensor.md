---
id: TASK-54.3
title: Compute the HDBSCAN distance matrix on the tfjs backend
status: To Do
assignee: []
created_date: '2026-06-24'
updated_date: '2026-06-24'
labels:
  - hdbscan
  - tfjs
  - performance
dependencies:
  - task-54.2
parent_task_id: TASK-54
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

Replace HDBSCAN's plain-JS float64 distance-matrix construction (`HDBSCAN.distance_matrix`, the `O(n²·d)` nested loops at `src/clustering/hdbscan.ts:142-163`) with the library's existing tfjs distance helper, producing the `(n, n)` distance matrix as a `Tensor2D` that stays on the backend for the downstream front-half stages.

This is the first front-half stage. Native `euclidean`/`manhattan` route through `pairwise_distance_matrix(points, metric)`; the `precomputed` case uploads the supplied square matrix to a `Tensor2D`. The result is **not** read back to JS in this subtask — it is handed to the next stage on-tensor (54.4 consumes it; the readback boundary is defined in 54.5).

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [ ] #1 The native euclidean/manhattan distance matrix is produced by `pairwise_distance_matrix` (reuse, not a reimplemented loop); the `precomputed` matrix is validated for squareness then uploaded to a `Tensor2D`
- [ ] #2 The distance matrix is returned/held as a `Tensor2D` for downstream on-tensor consumption; no per-cell JS metric branch remains in HDBSCAN
- [ ] #3 A comment documents the gram-matrix cancellation caveat for euclidean and the float32 clamp, and that labels are verified robust to it
- [ ] #4 `hdbscan.test.ts` passes under the 54.2 tolerances with labels exactly matching the oracle

<!-- AC:END -->

## Implementation Notes

Incremental: until 54.5 moves the readback to the MST boundary, this stage may `.array()` the tensor back so the existing JS core-distance/mreach steps still run — that keeps tests green but yields no perf benefit yet (expected). Watch the gram-matrix cancellation: if any fixture's labels flip versus the oracle, swap the euclidean path to a direct (non-gram) tensor distance for HDBSCAN rather than loosening label parity.
