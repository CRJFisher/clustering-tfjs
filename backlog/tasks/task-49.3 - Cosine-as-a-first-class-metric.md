---
id: task-49.3
title: Cosine as a first-class metric across clustering estimators and validation metrics
status: To Do
assignee: []
created_date: '2026-06-05'
labels:
  - tdt
dependencies: []
parent_task_id: task-49
---

## Description

Cosine geometry is a first-class option wherever distance or affinity drives a clustering or validation result. KMeans clusters direction-dominated data by treating L2-normalized vectors as points on the unit sphere; SpectralClustering builds its similarity graph from a cosine affinity; AgglomerativeClustering honors cosine linkage; and the internal-validation metrics score partitions under a caller-chosen distance metric. This lets users cluster and evaluate data where magnitude is uninformative and direction carries the signal, such as text embeddings, TF-IDF vectors, and high-dimensional sparse representations. Distance and affinity computations route through the shared `pairwise_distance_matrix` and graph affinity builders, keeping a single source of truth for each metric across the whole pipeline.

## Acceptance Criteria

- [ ] `compute_cosine_affinity` is implemented in `src/graph/affinity.ts` and reachable through the `compute_affinity_matrix` dispatcher when affinity is `'cosine'`, with colocated unit coverage in `src/graph/affinity.test.ts`
- [ ] `KMeansParams` in `src/clustering/types.ts` exposes `metric` with values `'euclidean' | 'cosine'`; when `metric` is `'cosine'`, `src/clustering/kmeans.ts` L2-normalizes the data and runs k-means++ seeding and Lloyd assignment through `pairwise_distance_matrix(points, metric)` with no inlined squared-euclidean distance on the cosine path
- [ ] `SpectralClusteringParams.affinity` in `src/clustering/types.ts` accepts `'cosine'` and `src/clustering/spectral.ts` produces a valid spectral embedding and `labels_` for `affinity='cosine'` on a fixture dataset
- [ ] AgglomerativeClustering with `metric='cosine'` (linkage one of `'complete' | 'average' | 'single'`) produces `labels_` matching sklearn AgglomerativeClustering up to label permutation, verified by a colocated test in `src/clustering/agglomerative.test.ts` against an `__fixtures__/agglomerative` cosine reference
- [ ] `silhouette_samples`, `silhouette_score`, `silhouette_score_subset` (`src/validation/silhouette.ts`) and `davies_bouldin`, `davies_bouldin_efficient` (`src/validation/davies_bouldin.ts`) accept a `metric` parameter selecting `'euclidean' | 'cosine'`; `calinski_harabasz` and `calinski_harabasz_efficient` (`src/validation/calinski_harabasz.ts`) are documented as variance-based and metric-independent
- [ ] every internal caller of the affected validation metrics passes `metric` explicitly with no default-argument shim: `src/clustering/spectral_optimization.ts` and `src/model_selection/find_optimal_clusters.ts`, plus the re-export barrels `src/validation/index.ts` and `src/index.ts`
- [ ] a standalone cosine pairwise-distance fixture generated from `sklearn.metrics.pairwise_distances(metric='cosine')` is added under `__fixtures__/` and asserted against `pairwise_distance_matrix(points, 'cosine')` in `src/distance/pairwise_distance.test.ts`
- [ ] no unsafe type assertions (`as any` / `as unknown` / `as never`) are introduced and the lint suite passes
