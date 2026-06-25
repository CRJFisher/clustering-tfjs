---
id: TASK-54.4
title: Compute HDBSCAN core distances on-tensor via tf.topk
status: Done
assignee: []
created_date: '2026-06-24'
updated_date: '2026-06-24'
labels:
  - hdbscan
  - tfjs
  - performance
dependencies:
  - task-54.3
parent_task_id: TASK-54
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

Replace HDBSCAN's core (k-)distance computation — the full per-row sort `D.map(row => [...row].sort())` at `src/clustering/hdbscan.ts:198` followed by `kdistance` reading column `k-1` — with an on-tensor `tf.topk` over the distance tensor from 54.3. The k-th smallest value is a pure order statistic, so the result is identical to the sort modulo float32, while running on the backend and avoiding the transient `n×n` sorted-copy allocation.

Mirror the established smallest-k idiom in `src/graph/affinity.ts:165` (`tf.topk(neg_dists, top_k)` on the negated row to select smallest). Core distances stay on-tensor for the mutual-reachability stage (54.5).

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [ ] #1 Core distances are computed with `tf.topk` (smallest `min_samples`) over the distance tensor, taking the `min_samples-1` order statistic per row; the result is a tensor, not read back in this subtask
- [ ] #2 The full per-row sort and its `n×n` sorted-copy allocation at `hdbscan.ts:198` are removed
- [ ] #3 The `self counts as first neighbour` semantics are preserved (diagonal 0 ⇒ `min_samples=1` yields 0), matching the previous core-distance definition
- [ ] #4 `hdbscan.test.ts` passes under the 54.2 tolerances with labels exactly matching the oracle

<!-- AC:END -->

## Implementation Notes

## High-level summary

HDBSCAN's second front-half stage now runs on the TensorFlow.js backend. The per-row JS sort (`D.map(row => [...row].sort(...))`) and `kdistance` call — which allocated an `O(n²)` sorted copy of the full distance matrix — are replaced by an on-tensor `tf.topk` order statistic over the negated `D_tensor`.

A private `core_distances(D_tensor, min_samples)` method encapsulates the computation: negating `D_tensor` promotes the `min_samples` smallest distances per row to `tf.topk`'s top slots (largest-first), column `min_samples−1` of the result is the negated k-th order statistic, and negating back recovers the core distance. The diagonal zero (self-distance) sorts highest in the negated space, so `min_samples=1` yields all-zero core distances — preserving the "self counts as first neighbour" convention.

The `kdistance` import is removed from `hdbscan.ts` (now dead in the production path); `kdistance.ts` itself is retained for its own tests and as a reference, with final cleanup deferred to task-54.10.

**Acceptance criteria addressed:**

- **AC#1** — Core distances are computed with `tf.topk` over `D_tensor`; `core_distances` returns a `Tensor1D`. Both `D_tensor` and `core_tensor` are read back to JS for the mutual-reachability tail (the current tensor/JS boundary); 54.5 moves that boundary to the MST input.
- **AC#2** — The `D.map(row => [...row].sort(...))` allocation and the `kdistance` call are removed.
- **AC#3** — Self-as-first-neighbour semantics preserved: `min_samples=1` slices column 0 of the topk result, which holds the negated self-distance (0), giving core distance 0.
- **AC#4** — `hdbscan.test.ts` passes 58/58 under the task-54.2 tolerances with labels exactly matching the oracle across all 33 fixtures.

**Tensor ownership.** `core_tensor` is hoisted before the try block so the `finally` can guard it: `core_tensor?.dispose()` runs on every exit path, even if `core_tensor.array()` rejects. The `tf.tidy` inside `core_distances` disposes all intermediates (the negated matrix, topk `values` and `indices`, the intermediate slice) and returns a single caller-owned `Tensor1D`.

**Review.** A 10-lens review (correctness ×3, data ×1, completeness ×2, IA ×3, adversarial cold-read) found no correctness defects. Three comment-quality fixes were applied: task-number references stripped from inline comments, "Incremental / task-54.5 will…" future-state framing replaced with a description of the current JS-tail readback, and an annotation added to the `as tf.Tensor1D` cast explaining tf.tidy's type erasure. The memory-regression gap (no HDBSCAN entry in `memory_regression.test.ts`) is tracked for task-54.8.

The standalone JS `kdistance` helper (`src/distance/kdistance.ts`) and its callers may now be unused by the production path; do not delete it here — its removal (and that of any other dead JS front-half code) is the YAGNI-gated cleanup in 54.10, after the benchmark decides whether a small-`n` JS fallback is retained. Do NOT keep both a JS-sort and a topk path behind a permanent toggle (NO BACKWARDS COMPATIBILITY) — any size gate is a separate, benchmark-justified decision.
