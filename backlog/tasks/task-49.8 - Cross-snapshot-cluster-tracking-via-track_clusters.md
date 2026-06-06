---
id: TASK-49.8
title: Cross-snapshot cluster tracking via track_clusters
status: Done
assignee: []
created_date: '2026-06-05'
updated_date: '2026-06-06 12:22'
labels:
  - tdt
dependencies:
  - task-49.3
  - task-49.6
  - task-49.7
parent_task_id: TASK-49
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clustering runs over evolving data produce a fresh set of clusters at each snapshot, but those clusters carry no identity across time, so callers cannot tell which cluster in the current snapshot continues, splits from, or merges into a cluster in the previous snapshot. `track_clusters` compares two consecutive snapshots by their cluster representative vectors, computes an optimal bipartite assignment between them, and classifies each cluster's fate as a transition (persist, emerge, die, merge, split) with a stable lifeline identity carried forward across snapshots. This gives callers a temporal view of cluster lifecycles for drifting and streaming data. The function is stateless: the caller owns and threads the tracking state, so tracking composes cleanly with any clustering pipeline without retaining hidden global state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `track_clusters(prev, curr, options, prev_state?)` is exported from `src/clustering/cluster_tracking.ts` and returns a `TrackingResult` the caller owns; no module-level mutable state is retained between calls
- [x] #2 a cost matrix between previous and current clusters is built from cosine distance over their representative vectors using `pairwise_distance_matrix` from `src/distance/pairwise_distance.ts` with metric `'cosine'` (no inline re-implementation)
- [x] #3 a pure-TypeScript Hungarian (linear-sum-assignment) solver computes the minimum-cost bipartite matching, with assignments whose cost exceeds `1 - threshold` pruned so dissimilar clusters are not matched
- [x] #4 each cluster is classified as one of `PERSIST`, `EMERGE`, `DIE`, `MERGE`, or `SPLIT` based on the matching, and the transitions are emitted in the `TrackingResult`
- [x] #5 lifeline identifiers are stable across snapshots: when `prev_state` is threaded in, a persisting cluster keeps its prior lifeline id and new clusters receive fresh ids
- [x] #6 rectangular cases where the previous and current cluster counts differ are handled without error and the asymmetric-matching behavior is documented
- [x] #7 `tools/sklearn_fixtures/generate_tracking.py` emits synthetic drifting-snapshot fixtures with reference assignments from scipy `linear_sum_assignment` into `__fixtures__/tracking/`, and a colocated test `src/clustering/cluster_tracking.test.ts` asserts the TypeScript assignment and transition labels match the reference
- [x] #8 the public barrel `src/index.ts` exports `track_clusters` and its `TrackingResult` type, and all call sites use it directly with no compatibility shims, aliases, or unsafe type casts
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
track_clusters: cosine cost via pairwise_distance_matrix, pure-TS Hungarian (linear_sum_assignment), threshold pruning, PERSIST/EMERGE/DIE/MERGE/SPLIT, stable lifeline ids, stateless. Rectangular handled. scipy reference fixtures (generate_tracking.py); exported from index.
<!-- SECTION:NOTES:END -->
