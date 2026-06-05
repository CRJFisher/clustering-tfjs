---
id: task-49.2
title: Define noise-label (-1) semantics and make validation metrics noise-aware
status: To Do
assignee: []
created_date: '2026-06-05'
labels:
  - tdt
dependencies: []
parent_task_id: task-49
---

## Description

Cluster labels carry a single authoritative meaning across the library: non-density estimators emit a dense labeling in `0..n_clusters-1`, and density estimators emit `-1` to mark a sample as noise. This shared contract lets density and partition estimators interoperate through the same `LabelVector` type and lets downstream consumers interpret `-1` consistently as "not assigned to any cluster."

Internal-validation metrics measure cohesion and separation of genuine clusters, so noise samples must not participate in their distance computations. `silhouette`, `calinski_harabasz`, and `davies_bouldin` therefore exclude `-1`-labeled samples before computing any pairwise distance or dispersion. They also remain numerically well-defined at the degenerate boundaries this introduces — input that is entirely noise, or that contains a single real cluster surrounded by noise — returning a defined result with no division by zero.

The label-semantics contract is recorded as a decision record so the meaning of `-1` is discoverable and stable for every estimator and metric that depends on it.

## Acceptance Criteria

- [ ] A decision record in `backlog/decisions/` states the cluster-label contract: non-density estimators emit dense labels `0..n_clusters-1` and density estimators emit `-1` for noise
- [ ] `silhouette_samples`, `silhouette_score`, and `silhouette_score_subset` in `src/validation/silhouette.ts` exclude `-1`-labeled samples before computing distances
- [ ] `calinski_harabasz` and `calinski_harabasz_efficient` in `src/validation/calinski_harabasz.ts` exclude `-1`-labeled samples before computing sum-of-squares
- [ ] `davies_bouldin` and `davies_bouldin_efficient` in `src/validation/davies_bouldin.ts` exclude `-1`-labeled samples before computing centroid and dispersion distances
- [ ] All six metric functions return a defined value with no division by zero when every label is `-1`
- [ ] All six metric functions return a defined value with no division by zero when input contains exactly one cluster plus `-1` noise samples
- [ ] Colocated tests in `src/validation/silhouette.test.ts`, `src/validation/calinski_harabasz.test.ts`, and `src/validation/davies_bouldin.test.ts` cover the all-noise and single-cluster-plus-noise edge cases
- [ ] Noise filtering is implemented in the metric computations themselves with no compatibility shim, wrapper, or alias bridging filtered and unfiltered call paths
- [ ] No unsafe type assertions (`as any`, `as unknown`, `as never`) are introduced
