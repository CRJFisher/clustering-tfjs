---
id: TASK-49.9
title: Public PCA estimator in src/decomposition
status: Done
assignee: []
created_date: '2026-06-05'
updated_date: '2026-06-06 12:22'
labels:
  - tdt
  - deferred
dependencies: []
parent_task_id: TASK-49
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A public principal-component-analysis estimator gives consumers a dimensionality-reduction primitive that matches scikit-learn's `PCA` numerically. SOM `'linear'`/`'pca'` weight initialization relies on a power-iteration computation of the leading principal components of a covariance matrix; a standalone `PCA` class in a dedicated `src/decomposition/` domain folder exposes that same computation as a first-class, serializable, sklearn-validated capability. The estimator fits a mean-centered data matrix, computes its leading components via power iteration (the library's tensor backend has no native eigendecomposition), and projects data into the reduced component space. SOM initialization draws on this single shared implementation so map seeding and public reduction stay numerically identical.

This is a Phase 5 capability: build it when a consumer needs public PCA (for example, pre-projecting high-dimensional embeddings before density clustering).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A `PCA` class exists in `src/decomposition/pca.ts` exposing `fit(X)`, `transform(X)`, `fit_transform(X)`, and `inverse_transform(X)`, with fitted attributes `components_`, `explained_variance_`, and `mean_` populated by `fit`
- [x] #2 `PCA` accepts `n_components` and `random_state` (matching the `CoreClusteringParams.random_state` convention) so power-iteration seeding is deterministic for a fixed `random_state`
- [x] #3 `PCA` exposes `to_json()` and `from_json()` so a fitted estimator round-trips its `components_`, `explained_variance_`, `mean_`, and params without loss
- [x] #4 `fit` rejects `n_components` greater than the number of input features by throwing, validated by a colocated test in `src/decomposition/pca.test.ts`
- [x] #5 Calling `fit_predict` on `PCA` throws, since `PCA` is a dimensionality-reduction estimator and not a clusterer
- [x] #6 SOM `'linear'`/`'pca'` initialization in `src/clustering/som_neighborhood.ts` sources its principal components from the shared computation in `src/decomposition/pca.ts`, with no separate principal-component implementation in `som_neighborhood.ts`, and the SOM initialization fixtures in `__fixtures__/som/` still pass
- [x] #7 `tools/sklearn_fixtures/generate_pca.py` generates reference fixtures into `__fixtures__/pca/` using `sklearn.decomposition.PCA(svd_solver='full')`
- [x] #8 Colocated tests in `src/decomposition/pca.test.ts` compare `PCA.components_` and `PCA.transform(X)` against the `__fixtures__/pca/` references up to per-axis sign, applying the `svd_flip` sign-normalization convention
- [x] #9 `PCA` is exported from the public barrel `src/index.ts`
- [x] #10 Eigenvalue/variance accessors return values consistent with the sklearn reference, and no `as any`/`as unknown`/`as never` casts are introduced
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PCA in src/decomposition: fit/transform/fit_transform/inverse_transform, components_/explained_variance_/mean_, to_json/from_json, n_components>features throws, fit_predict throws. Shared power_iteration_eig also seeds SOM linear/pca init (no separate impl). sklearn svd_flip fixtures.
<!-- SECTION:NOTES:END -->
