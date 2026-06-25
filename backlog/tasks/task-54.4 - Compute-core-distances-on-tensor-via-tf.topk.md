---
id: TASK-54.4
title: Compute HDBSCAN core distances on-tensor via tf.topk
status: To Do
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

The standalone JS `kdistance` helper (`src/distance/kdistance.ts`) and its callers may now be unused by the production path; do not delete it here — its removal (and that of any other dead JS front-half code) is the YAGNI-gated cleanup in 54.10, after the benchmark decides whether a small-`n` JS fallback is retained. Do NOT keep both a JS-sort and a topk path behind a permanent toggle (NO BACKWARDS COMPATIBILITY) — any size gate is a separate, benchmark-justified decision.
