---
id: TASK-52.1
title: >-
  Fix Math.max spread RangeError in agglomerative compute_medoids for large
  datasets
status: Done
assignee:
  - crjfisher
created_date: '2026-06-10 08:55'
updated_date: '2026-06-10 10:03'
labels:
  - bug
  - confirmed
dependencies: []
references:
  - 'src/clustering/agglomerative.ts:257'
parent_task_id: TASK-52
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

AgglomerativeClustering.compute*medoids uses `Math.max(-1, ...this.labels*)` to find the number of clusters. Spreading a 200k+ element array as function arguments exhausts the JS engine's call-stack argument limit, throwing RangeError and making the method unusable on large datasets.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [x] #1 Math.max spread on labels\_ is replaced with a reduce or loop-based equivalent
- [x] #2 compute_medoids works correctly on datasets with 200k+ samples
- [x] #3 A regression test with at least 300,000 synthetic samples calls compute_medoids and does not throw (large-n smoke test)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->

1. Replace Math.max spread in compute*medoids with a loop that iterates labels* once.\n2. Add a 300k-sample regression test that bypasses fit() by setting labels\_ directly.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->

Replaced Math.max(-1, ...this.labels*) + 1 with a single forward loop in compute_medoids (agglomerative.ts:257-260). The loop initializes n_clusters=0 and updates it when a label >= current max is encountered, producing semantically identical output including the empty-array edge case. Added a 300k-sample smoke test in agglomerative.test.ts that bypasses fit() by injecting labels* directly (labels* is public per sklearn convention), avoiding the O(n²) agglomerative fit cost. Test asserts the call resolves and medoid_indices* has the correct cluster count. All 68 tests pass. Note: spectral.ts:238 and :605 contain the same Math.max/min spread pattern and carry the same RangeError risk on large inputs — filed as a follow-up.

<!-- SECTION:NOTES:END -->
