---
id: TASK-49.2
title: Define noise-label (-1) semantics and make validation metrics noise-aware
status: Done
assignee: []
created_date: '2026-06-05'
updated_date: '2026-06-06 12:22'
labels: []
dependencies: []
parent_task_id: TASK-49
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

Cluster labels carry a single authoritative meaning across the library: non-density estimators emit a dense labeling in `0..n_clusters-1`, and density estimators emit `-1` to mark a sample as noise. This shared contract lets density and partition estimators interoperate through the same `LabelVector` type and lets downstream consumers interpret `-1` consistently as "not assigned to any cluster."

Internal-validation metrics measure cohesion and separation of genuine clusters, so noise samples must not participate in their distance computations. `silhouette`, `calinski_harabasz`, and `davies_bouldin` therefore exclude `-1`-labeled samples before computing any pairwise distance or dispersion. They also remain numerically well-defined at the degenerate boundaries this introduces — input that is entirely noise, or that contains a single real cluster surrounded by noise — returning a defined result with no division by zero.

The label-semantics contract is recorded as a decision record so the meaning of `-1` is discoverable and stable for every estimator and metric that depends on it.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [x] #1 A decision record in `backlog/decisions/` states the cluster-label contract: non-density estimators emit dense labels `0..n_clusters-1` and density estimators emit `-1` for noise
- [x] #2 `silhouette_samples`, `silhouette_score`, and `silhouette_score_subset` in `src/validation/silhouette.ts` exclude `-1`-labeled samples before computing distances
- [x] #3 `calinski_harabasz` and `calinski_harabasz_efficient` in `src/validation/calinski_harabasz.ts` exclude `-1`-labeled samples before computing sum-of-squares
- [x] #4 `davies_bouldin` and `davies_bouldin_efficient` in `src/validation/davies_bouldin.ts` exclude `-1`-labeled samples before computing centroid and dispersion distances
- [x] #5 All six metric functions return a defined value with no division by zero when every label is `-1`
- [x] #6 All six metric functions return a defined value with no division by zero when input contains exactly one cluster plus `-1` noise samples
- [x] #7 Colocated tests in `src/validation/silhouette.test.ts`, `src/validation/calinski_harabasz.test.ts`, and `src/validation/davies_bouldin.test.ts` cover the all-noise and single-cluster-plus-noise edge cases
- [x] #8 Noise filtering is implemented in the metric computations themselves with no compatibility shim, wrapper, or alias bridging filtered and unfiltered call paths
- [x] #9 No unsafe type assertions (`as any`, `as unknown`, `as never`) are introduced
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->

decision-1 records the -1 noise contract. silhouette/calinski/davies (+ efficient/subset variants) filter -1 before distances; degenerate all-noise and single-cluster+noise return defined 0 (no div-by-zero); genuine <2-cluster (no noise) still throws. noise_filtered_indices helper.

<!-- SECTION:NOTES:END -->
