---
id: task-49.5
title: HDBSCAN density-based clustering estimator
status: To Do
assignee: []
created_date: '2026-06-05'
labels:
  - tdt
dependencies:
  - task-49.1
  - task-49.2
  - task-49.3
  - task-49.4
parent_task_id: task-49
---

## Description

Provide HDBSCAN as a density-based clustering estimator so callers can discover clusters of varying density without specifying a cluster count, and have points in sparse regions flagged as noise. The estimator derives a cluster hierarchy from mutual-reachability distances and a minimum spanning tree, condenses that hierarchy, and selects stable clusters. It exposes per-point membership strengths and, on request, representative exemplar points per cluster, matching scikit-learn's HDBSCAN behaviour so results are trustworthy against an established reference.

## Acceptance Criteria

- [ ] `HDBSCAN` in `src/clustering/hdbscan.ts` implements `BaseClustering<HDBSCANParams>` with `fit`, `fit_predict`, and `dispose`; it exposes no `predict` method and `HDBSCANParams` carries no `n_clusters` field
- [ ] `HDBSCANParams` in `src/clustering/types.ts` declares `min_cluster_size`, `min_samples`, `metric`, `cluster_selection_epsilon`, `cluster_selection_method`, and `store_exemplars`
- [ ] `metric` values `euclidean` and `manhattan` are computed natively from the input data matrix, and `metric` `precomputed` accepts an `(n, n)` distance matrix directly
- [ ] `labels_` assigns noise points the value `-1` consistently, and `probabilities_` holds per-point cluster membership strengths in `[0, 1]`
- [ ] `store_exemplars` set to true populates `exemplar_indices_` with representative point indices per selected cluster, and leaves it unpopulated otherwise
- [ ] `tools/sklearn_fixtures/generate_hdbscan.py` emits reference fixtures into `__fixtures__/hdbscan/` sweeping `min_cluster_size`, `min_samples`, `cluster_selection_epsilon`, and `cluster_selection_method`, including a cosine case supplied as a precomputed cosine distance matrix
- [ ] colocated `src/clustering/hdbscan.test.ts` loads the hdbscan fixtures and asserts `labels_` match scikit-learn up to label permutation with consistent `-1` noise, and `probabilities_` match within tolerance
- [ ] `HDBSCAN` is exported from `src/clustering/init.ts` and the public barrel `src/index.ts` alongside the other estimators
- [ ] `hdbscan.ts` consumes its density and tree primitives from their domain modules rather than reimplementing them inline: k-distance from `src/distance/kdistance.ts`, mutual reachability from `src/graph/mutual_reachability.ts`, the minimum spanning tree from `src/graph/minimum_spanning_tree.ts`, and the condensation tree from `src/graph/condensation_tree.ts`
