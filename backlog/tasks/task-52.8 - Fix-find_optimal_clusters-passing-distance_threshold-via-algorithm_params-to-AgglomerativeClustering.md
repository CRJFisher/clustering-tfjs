---
id: TASK-52.8
title: >-
  Fix find_optimal_clusters passing distance_threshold via algorithm_params to
  AgglomerativeClustering
status: To Do
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

- [ ] #1 algorithm_params for the agglomerative algorithm is validated to exclude distance_threshold (which is controlled by the k-sweep loop)
- [ ] #2 Passing distance_threshold via algorithm_params produces a clear error at the find_optimal_clusters boundary, not inside AgglomerativeClustering
- [ ] #3 A test passes distance_threshold via algorithm_params and asserts a clear error is thrown; a companion test passes other agglomerative params (linkage, metric) via algorithm_params and asserts they are accepted correctly (parameter interaction boundary test)
<!-- AC:END -->
