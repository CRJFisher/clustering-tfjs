---
id: task-12.5
title: Propagate randomState to all spectral helper utilities
status: To Do
assignee: []
created_date: '2025-07-18'
labels: [spectral, rng]
dependencies: [task-12]
---

## Description (the why)

`SpectralClustering.randomState` is currently passed to K-Means only.  Several helper functions still fall back to JavaScript’s global RNG when deterministic tie-breaking is required:

* Jacobi eigen-solver pivot selection when encountering equal off-diagonal magnitudes.
* k-NN affinity builder – deterministic ordering of equidistant neighbours in extremely small / duplicated datasets.
* Any remaining `Math.random` fallback inside `smallest_eigenvectors`.

This leaks non-determinism into the embedding and breaks exact parity with scikit fixtures.

## Acceptance Criteria (the what)

- [ ] All random choices inside spectral utilities are sourced from `make_random_stream(randomState)`.
- [ ] Public `SpectralClustering` constructor passes its seed down to every helper.
- [ ] Add regression test: running the same model twice with `randomState = 123` yields identical labels and embeddings.

## Implementation Plan (the how)

1. Thread an optional RNG into `compute_knn_affinity`, `jacobi_eigen_decomposition`, `smallest_eigenvectors`.
2. Default to non-deterministic `Math.random` only when no seed provided to the top-level API.

