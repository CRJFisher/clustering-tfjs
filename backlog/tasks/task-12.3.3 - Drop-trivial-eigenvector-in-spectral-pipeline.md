---
id: task-12.3.3
title: Drop trivial eigenvector in SpectralClustering pipeline
status: To Do
assignee: []
created_date: '2025-07-17'
labels: []
dependencies: [task-12.3.2]
---

## Description (the why)

The first eigenpair (eigen-value ≈ 0) corresponds to the constant eigenvector of the normalised Laplacian and must be removed before row-normalising the embedding.  scikit-learn discards this column.  Our current implementation still feeds it into k-means which harms clustering quality.

## Acceptance Criteria (the what)

- [ ] `SpectralClustering.fit()` requests `nClusters + 1` eigenvectors from `smallest_eigenvectors()` and slices away the first column **after** deterministic post-processing.
- [ ] Updated code passes robustness test `correctly separates two obvious blobs`.
- [ ] High-level docstring of `SpectralClustering.fit()` mentions the exclusion of the trivial component.
- [ ] All tensors created for the additional column are disposed.

## Implementation Plan (the how)

1. Modify call site in `src/clustering/spectral.ts`.
2. Use `tf.slice([0,1],[-1,nClusters])` then dispose the full matrix.
3. Re-run unit tests & fix any shape assumptions downstream.

## Dependencies

- Requires tasks 12.3.1 and 12.3.2 so that ordering/sign-fixed matrix with trivial component is available.

## Implementation Notes

Initial implementation (2025-07-17):

• SpectralClustering.fit() now requests k+1 eigenvectors via smallest_eigenvectors, slices away the first column using tf.slice, and disposes the temporary U_full.
• Code compiles and other unit tests pass; reference-parity fixtures improve but still below 0.95 and robustness ‘two blobs’ test fails – indicates further adjustments required (possibly normalisation or sign alignment).

Next steps: deeper debug with upcoming task-12.3.5 to inspect embeddings; ensure sign convention consistent after slicing.
