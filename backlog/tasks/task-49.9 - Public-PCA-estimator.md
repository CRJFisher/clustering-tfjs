---
id: task-49.9
title: Public PCA estimator in src/decomposition
status: To Do
assignee: []
created_date: '2026-06-05'
labels:
  - tdt
  - deferred
dependencies: []
parent_task_id: task-49
---

## Description

A public principal-component-analysis estimator gives consumers a dimensionality-reduction primitive that matches scikit-learn's `PCA` numerically. SOM `'linear'`/`'pca'` weight initialization relies on a power-iteration computation of the leading principal components of a covariance matrix; a standalone `PCA` class in a dedicated `src/decomposition/` domain folder exposes that same computation as a first-class, serializable, sklearn-validated capability. The estimator fits a mean-centered data matrix, computes its leading components via power iteration (the library's tensor backend has no native eigendecomposition), and projects data into the reduced component space. SOM initialization draws on this single shared implementation so map seeding and public reduction stay numerically identical.

This is a Phase 5 capability: build it when a consumer needs public PCA (for example, pre-projecting high-dimensional embeddings before density clustering).

## Acceptance Criteria

- [ ] A `PCA` class exists in `src/decomposition/pca.ts` exposing `fit(X)`, `transform(X)`, `fit_transform(X)`, and `inverse_transform(X)`, with fitted attributes `components_`, `explained_variance_`, and `mean_` populated by `fit`
- [ ] `PCA` accepts `n_components` and `random_state` (matching the `CoreClusteringParams.random_state` convention) so power-iteration seeding is deterministic for a fixed `random_state`
- [ ] `PCA` exposes `to_json()` and `from_json()` so a fitted estimator round-trips its `components_`, `explained_variance_`, `mean_`, and params without loss
- [ ] `fit` rejects `n_components` greater than the number of input features by throwing, validated by a colocated test in `src/decomposition/pca.test.ts`
- [ ] Calling `fit_predict` on `PCA` throws, since `PCA` is a dimensionality-reduction estimator and not a clusterer
- [ ] SOM `'linear'`/`'pca'` initialization in `src/clustering/som_neighborhood.ts` sources its principal components from the shared computation in `src/decomposition/pca.ts`, with no separate principal-component implementation in `som_neighborhood.ts`, and the SOM initialization fixtures in `__fixtures__/som/` still pass
- [ ] `tools/sklearn_fixtures/generate_pca.py` generates reference fixtures into `__fixtures__/pca/` using `sklearn.decomposition.PCA(svd_solver='full')`
- [ ] Colocated tests in `src/decomposition/pca.test.ts` compare `PCA.components_` and `PCA.transform(X)` against the `__fixtures__/pca/` references up to per-axis sign, applying the `svd_flip` sign-normalization convention
- [ ] `PCA` is exported from the public barrel `src/index.ts`
- [ ] Eigenvalue/variance accessors return values consistent with the sklearn reference, and no `as any`/`as unknown`/`as never` casts are introduced
