---
id: TASK-52.8
title: >-
  Fix find_optimal_clusters passing distance_threshold via algorithm_params to
  AgglomerativeClustering
status: Done
assignee: []
created_date: '2026-06-10 08:56'
labels:
  - bug
  - plausible
dependencies: []
references:
  - 'src/model_selection/find_optimal_clusters.ts:276'
  - 'src/clustering/agglomerative.ts:95'
parent_task_id: TASK-52
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

find_optimal_clusters constructs AgglomerativeClustering with {n_clusters: k, ...algorithm_params} where algorithm_params is typed as Record&lt;string, unknown&gt;. If a caller passes algorithm_params: {distance_threshold: 2.5}, the spread produces both n_clusters and distance_threshold, triggering the new validate_params runtime error 'Provide exactly one of n_clusters or distance_threshold'. TypeScript cannot catch this because of the untyped spread.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [x] #1 algorithm_params for the agglomerative algorithm is validated to exclude distance_threshold (which is controlled by the k-sweep loop)
- [x] #2 Passing distance_threshold via algorithm_params produces a clear error at the find_optimal_clusters boundary, not inside AgglomerativeClustering
- [x] #3 A test passes distance_threshold via algorithm_params and asserts a clear error is thrown; a companion test passes other agglomerative params (linkage, metric) via algorithm_params and asserts they are accepted correctly (parameter interaction boundary test)
<!-- AC:END -->

## Implementation Notes

## High-level summary

`find_optimal_clusters` sweeps candidate k values from `min_clusters` to `max_clusters`, constructing an `AgglomerativeClustering` instance for each k with `{ n_clusters: k, ...algorithm_params }`. If a caller placed `distance_threshold` in `algorithm_params`, the spread would produce an object with both `n_clusters` and `distance_threshold` set, triggering `AgglomerativeClustering`'s mutual-exclusion check deep inside the k-sweep loop — once per iteration — with an error message that gives no hint about the outer orchestrator.

The fix adds a single guard at the `find_optimal_clusters` input-validation block, before tensor creation and the sweep loop begins. It checks `algorithm === 'agglomerative' && 'distance_threshold' in algorithm_params` and throws immediately with a message that names the prohibited key, explains why (the loop owns the stopping criterion), and points the caller toward `min_clusters`/`max_clusters`. No changes to `AgglomerativeClustering` or its types were needed — the invariant is correctly enforced there; this is purely a boundary-error-quality fix.

Two tests were added: one that passes `{ distance_threshold: 2.5 }` and asserts the error fires at the `find_optimal_clusters` boundary (not during clustering), and one that passes `{ linkage: 'complete', metric: 'euclidean' }` and asserts those accepted params produce a valid result. The `algorithm_params` JSDoc and `docs/API.md` option table were updated to surface the new constraint at the API surface, making it discoverable before a caller hits a runtime error.
