---
id: TASK-49.6
title: KMeans nearest-centroid predict and centroid accessor
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

KMeans learns cluster centroids during fitting but offers no way to assign new, unseen points to those clusters or to read the centroids back in a plain numeric form. A nearest-centroid `predict(X)` lets callers label held-out data using a fitted model, and a `get_centroids()` accessor exposes the learned centroids as a plain `number[][]` without forcing callers to reach into tensor internals. Together these complete the estimator's inference surface and bring it to parity with scikit-learn's `KMeans.predict`, so downstream consumers and validation fixtures can compare assignments directly against the reference implementation.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [x] #1 `KMeans.predict(X)` assigns each row of `X` to its nearest centroid using `pairwise_distance_matrix` from `src/distance/pairwise_distance.ts` and returns the resulting labels as `number[]`
- [x] #2 `KMeans.predict(X)` throws a descriptive error when called before `fit()` has populated `centroids_`, with no silent fallback or default behavior
- [x] #3 `get_centroids()` returns the learned centroids as `number[][]` (shape `n_clusters x n_features`) and throws when the model is unfitted
- [x] #4 `predict` and `get_centroids` both read the fitted `centroids_` attribute (`tf.Tensor2D`, sklearn-style trailing-underscore naming) as the single source of truth for centroid values
- [x] #5 A dedicated sklearn fixture generator `tools/sklearn_fixtures/generate_kmeans.py` (matching the per-algorithm pattern of `generate_spectral.py`/`generate_som.py`) emits `cluster_centers_` and the labels produced by sklearn `KMeans.predict` on a held-out sample set into `__fixtures__/kmeans/`
- [x] #6 Colocated test `src/clustering/kmeans.test.ts` asserts `get_centroids()` matches the fixture `cluster_centers_` up to cluster permutation within a tolerance, and that `predict()` on held-out data reproduces the sklearn predict labels up to the same permutation (exact cluster-id equality is not attainable cross-implementation because k-means++ seeding uses a different RNG; the `from_json` predict-after-restore test asserts exact within-implementation equality)
- [x] #7 No type assertions (`as any` / `as unknown` / `as never`) and no compatibility shim or wrapper are introduced
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->

KMeans.predict(X) (nearest centroid via pairwise*distance_matrix, metric-aware) and get_centroids(); throw before fit. sklearn cluster_centers* + predict fixtures (generate_kmeans.py).

<!-- SECTION:NOTES:END -->
