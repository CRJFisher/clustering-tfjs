---
id: task-12.1
title: Drop all trivial eigenvectors when multiple zero-eigenvalues occur
status: To Do
assignee: []
created_date: '2025-07-18'
labels: [spectral]
dependencies: [task-12]
---

## Description (the why)

If the affinity graph has *c* connected components, the normalised Laplacian has **c** independent eigen-vectors with eigen-value 0.  scikit-learn discards *all* these constant vectors before row-normalising the spectral embedding.  Our current implementation only drops the **first** column.  When *c > 1* an additional trivial column remains and degrades the subsequent k-means step – clearly visible in the failing reference fixtures with three clusters.

## Acceptance Criteria (the what)

- [ ] `smallest_eigenvectors()` returns *k + c* columns where *c* is the number of zero eigen-values detected within numerical tolerance (≤ 1e-10).
- [ ] `SpectralClustering.fit()` drops **all** columns whose corresponding eigen-value is ≤ 1e-10 before the row-normalisation step.
- [ ] Unit test: a block-diagonal affinity matrix with two disconnected components leads to an embedding that contains **no** constant columns.
- [ ] All intermediate tensors for the discarded components are disposed.

## Implementation Plan (the how)

1. Extend `smallest_eigenvectors()` to return eigen-values alongside vectors.
2. Count #values ≈ 0 and slice them away in `spectral.ts`.
3. Ensure resulting matrix still has exactly `nClusters` columns; if not enough informative vectors remain, throw an explanatory error (mirrors scikit behaviour).
4. Add focussed unit test under `test/unit/spectral_trivial_vectors.test.ts`.

