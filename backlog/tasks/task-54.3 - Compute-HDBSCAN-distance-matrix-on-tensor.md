---
id: TASK-54.3
title: Compute the HDBSCAN distance matrix on the tfjs backend
status: Done
assignee: []
created_date: '2026-06-24'
updated_date: '2026-06-25'
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

- [x] #1 The native euclidean/manhattan distance matrix is produced by `pairwise_distance_matrix` (reuse, not a reimplemented loop); the `precomputed` matrix is validated for squareness then uploaded to a `Tensor2D`
- [x] #2 The distance matrix is returned/held as a `Tensor2D` for downstream on-tensor consumption; no per-cell JS metric branch remains in HDBSCAN
- [x] #3 A comment documents the gram-matrix cancellation caveat for euclidean and the float32 clamp, and that labels are verified robust to it
- [x] #4 `hdbscan.test.ts` passes under the 54.2 tolerances with labels exactly matching the oracle

<!-- AC:END -->

## Implementation Notes

## High-level summary

HDBSCAN's first front-half stage now runs on the TensorFlow.js backend. `HDBSCAN.distance_matrix` previously built the dense `(n, n)` matrix with an `O(n²·d)` plain-JS float64 nested loop and a per-cell `euclidean`/`manhattan` branch; it now returns a `Tensor2D`. Native metrics are delegated to the library's existing `pairwise_distance_matrix` (the same helper spectral affinity and cosine k-means seeding use), and a `precomputed` matrix is validated for squareness and uploaded to a tensor. The hand-rolled distance loop is deleted, so there is no longer any per-cell distance arithmetic in the estimator.

This subtask is deliberately incremental: the matrix is produced on-tensor, but `fit` immediately reads it back to a `number[][]` with a single `.array()` so the unchanged float64 tail (core distances → mutual reachability → MST → condensed tree) still runs. That keeps the suite green without yet realizing a speedup — task-54.5 moves the readback to the MST boundary, at which point the front-half stays on the backend end-to-end and the win is realized. The migration's only observable effect today is that distances are computed in float32 rather than float64.

**Acceptance criteria addressed:**

- **AC#1** — Native `euclidean`/`manhattan` route through `pairwise_distance_matrix(points, metric)` (reuse, no reimplemented loop). After the `precomputed` branch returns, TypeScript narrows `metric` to `'euclidean' | 'manhattan'`, so the call typechecks with no cast. The `precomputed` path validates squareness — `shape[0] === shape[1]` for tensor input, per-row length for array input — before uploading to a `Tensor2D`.
- **AC#2** — `distance_matrix` returns `tf.Tensor2D`; the nested per-cell metric loop and its `manhattan`/euclidean branch are gone entirely.
- **AC#3** — An inline comment at the native call site documents the gram-matrix cancellation caveat (`‖x‖²+‖y‖²−2·xᵀy` can produce tiny negative squared distances), the float32 `maximum(·, 0)` clamp inside the helper that pins them to zero, and that labels are verified robust to the resulting drift against the oracle under the task-54.2 tolerances.
- **AC#4** — `hdbscan.test.ts` passes 58/58 under the task-54.2 tolerances. Beyond the suite, every one of the 33 fixtures was checked directly against the task-54.1 float64 oracle: **0 label mismatches** (strict equivalence up to cluster-id permutation, `-1` noise fixed), with worst-case per-point probability drift of **9.9e-5** — exactly the figure the task-54 migration probe measured, and well inside the `TIE_FREE_PROB_ATOL = 1e-3` bound. The gram-matrix fallback to a direct (non-gram) euclidean distance was therefore not needed.

**Tensor ownership.** `distance_matrix` always returns a freshly-owned tensor: the native paths return a new matrix from `pairwise_distance_matrix` (which never disposes its input), the precomputed-array path returns `tf.tensor2d(rows)`, and the precomputed-**tensor** path returns `matrix.clone()` so a caller-supplied tensor is never disposed by the estimator. `fit` wraps its body in `try/finally` and disposes the matrix exactly once on every path — normal completion, the `n === 1` early return, and any throw in the JS tail. The native-array branch disposes its transient `points` tensor in its own `try/finally`. Empty input is rejected before any tensor allocation (so `tf.tensor2d([])` is never reached) and before `dispose()`, preserving the contract that a failed re-fit leaves prior fitted state intact.

**Review.** A multi-lens review (correctness ×2, contracts, completeness, IA, adversarial cold-read) found no correctness defects — the contracts lens returned only confidence-100 affirmations of the type narrowing, `.clone()` return type, sync-in-async rejection, and single-dispose discipline. Two documentation-accuracy fixes were applied: the method JSDoc no longer claims the matrix "stays on the backend for downstream stages" (untrue until 54.5), and the gram-matrix comment is scoped explicitly to the euclidean metric. A suggestion to add inline comments duplicating the JSDoc's ownership explanation was declined as surplus.

Watch item carried forward: if a future fixture's labels ever flip versus the oracle, swap the euclidean path to a direct (non-gram) tensor distance for HDBSCAN rather than loosening label parity.
