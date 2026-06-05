---
id: task-49.6
title: KMeans nearest-centroid predict and centroid accessor
status: To Do
assignee: []
created_date: '2026-06-05'
labels:
  - tdt
dependencies: []
parent_task_id: task-49
---

## Description

KMeans learns cluster centroids during fitting but offers no way to assign new, unseen points to those clusters or to read the centroids back in a plain numeric form. A nearest-centroid `predict(X)` lets callers label held-out data using a fitted model, and a `get_centroids()` accessor exposes the learned centroids as a plain `number[][]` without forcing callers to reach into tensor internals. Together these complete the estimator's inference surface and bring it to parity with scikit-learn's `KMeans.predict`, so downstream consumers and validation fixtures can compare assignments directly against the reference implementation.

## Acceptance Criteria

- [ ] `KMeans.predict(X)` assigns each row of `X` to its nearest centroid using `pairwise_distance_matrix` from `src/distance/pairwise_distance.ts` and returns the resulting labels as `number[]`
- [ ] `KMeans.predict(X)` throws a descriptive error when called before `fit()` has populated `centroids_`, with no silent fallback or default behavior
- [ ] `get_centroids()` returns the learned centroids as `number[][]` (shape `n_clusters x n_features`) and throws when the model is unfitted
- [ ] `predict` and `get_centroids` both read the fitted `centroids_` attribute (`tf.Tensor2D`, sklearn-style trailing-underscore naming) as the single source of truth for centroid values
- [ ] `tools/sklearn_fixtures/generate.py` emits `cluster_centers_` and the labels produced by sklearn `KMeans.predict` on a held-out sample set into the fixture JSON under `__fixtures__/`
- [ ] Colocated test `src/clustering/kmeans.test.ts` asserts `get_centroids()` matches the fixture `cluster_centers_` within `rtol=1e-5` and that `predict()` on held-out data reproduces the sklearn predict labels exactly
- [ ] No type assertions (`as any` / `as unknown` / `as never`) and no compatibility shim or wrapper are introduced
