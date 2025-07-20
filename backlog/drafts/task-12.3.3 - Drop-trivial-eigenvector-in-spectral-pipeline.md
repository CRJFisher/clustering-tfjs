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

---

Progress update (2025-07-18):

• Discovered that requesting `k + 1` eigenvectors can exceed the matrix
  dimension for very small input datasets (e.g. `nSamples ≤ k`).  This led
  to out-of-bounds `tf.slice` calls which surfaced as `Invalid TF_Status`
  errors in unit test *spectral_ninit_default*.

  → Patched `src/utils/laplacian.ts` so that
  `smallest_eigenvectors()` now caps the returned column count at `n`.

• Refined the *trivial eigenvector removal* logic inside
  `src/clustering/spectral.ts`:

  1. Added robust slicing that only drops column 0 when another column is
     actually present.
  2. Implemented automatic zero-padding when fewer than `nClusters`
     informative components are available so that downstream K-Means always
     receives an embedding with the expected dimensionality.

  This fixed the TF slicing runtime error and all `spectral_ninit_default`
  tests now pass.

• Confirmed that no NaNs are produced in the embeddings for the reference
  fixtures (`debug_fixture.ts` helper).  However, ARI scores are still below
  the 0.95 threshold for several datasets and the “two obvious blobs”
  robustness test remains red.

• Preliminary investigation suggests that row-normalisation (step 4) and/or
  sign consistency after zero-padding may require further tweaks.  These
  will be explored in task 12.3.5 together with intermediate tensor dumps.
