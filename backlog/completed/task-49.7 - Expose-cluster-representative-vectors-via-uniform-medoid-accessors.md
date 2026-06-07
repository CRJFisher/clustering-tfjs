---
id: TASK-49.7
title: Expose cluster representative vectors via uniform medoid accessors
status: Done
assignee: []
created_date: '2026-06-05'
updated_date: '2026-06-06 12:22'
labels:
  - tdt
dependencies:
  - task-49.6
parent_task_id: TASK-49
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

Consumers need a uniform way to obtain a representative sample for each cluster across estimators. KMeans exposes synthetic `centroids_`, but AgglomerativeClustering and SpectralClustering produce only label assignments, leaving callers without a representative vector per cluster. A single `ClusterRepresentations` contract and a shared `select_medoids` utility give every estimator a consistent, sklearn-style way to report which actual data samples best represent each cluster, so downstream code (summarization, labelling, nearest-representative lookups) works identically regardless of which algorithm produced the labels.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [x] #1 `src/clustering/representations.ts` defines the `ClusterRepresentations` interface with optional fields `centroids_`, `medoid_indices_`, and `exemplar_indices_`, typed against `DataMatrix`/`LabelVector` from `src/clustering/types.ts`
- [x] #2 `src/clustering/medoid_selection.ts` exports `select_medoids(X, labels, n_clusters, metric)` that returns, for each cluster, the index of the sample closest to that cluster's mean under the requested metric
- [x] #3 `select_medoids` runs in `O(n*k)` over `n` samples and `k` clusters, computing each cluster mean once and a single closest-sample pass, without materialising a full `n`-by-`n` pairwise distance matrix
- [x] #4 `select_medoids` handles empty and sparse label sets defensively: a cluster with no assigned samples yields no medoid index rather than throwing or returning a fabricated index, and the result correctly maps medoid positions back to original sample indices
- [x] #5 AgglomerativeClustering exposes a `medoid_indices_` fitted attribute and a `compute_medoids(X)` method that populates it via `select_medoids` using the estimator's `labels_`; SpectralClustering exposes representative vectors through the same `ClusterRepresentations` surface
- [x] #6 `tools/sklearn_fixtures/generate_agglomerative_medoids.py` generates post-hoc medoid reference fixtures into `__fixtures__/agglomerative/`, and stores per-sample distances to the cluster mean alongside each expected medoid index so tie cases are disambiguated
- [x] #7 a colocated test (`src/clustering/medoid_selection.test.ts`) asserts exact medoid index identity against the generated fixtures, using the stored distances to resolve ties deterministically
- [x] #8 metric selection threads through `select_medoids` without a default-bridging shim: every call site passes the metric explicitly and distance computation reuses `src/distance/pairwise_distance.ts` metric handling rather than a duplicated inline implementation
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->

ClusterRepresentations interface; select*medoids (O(n\*d), reuses tensor_ops metrics, -1 for empty clusters); Agglomerative/Spectral compute_medoids + medoid_indices*. generate_agglomerative_medoids.py fixtures with per-sample distances for tie resolution.

<!-- SECTION:NOTES:END -->
