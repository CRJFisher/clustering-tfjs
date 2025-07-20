---
id: task-12.4
title: Align K-Means empty-cluster handling and tie-breaking with scikit-learn
status: To Do
assignee: []
created_date: '2025-07-18'
labels: [kmeans]
dependencies: [task-11]
---

## Description (the why)

Our internal K-Means helper deviates from scikit-learn in two places that affect deterministic parity and sometimes produce NaNs / singleton clusters:

1. **Empty cluster reseeding** – we pick a random sample, sklearn picks the point with the largest potential (furthest from its closest centroid).
2. **k-means++ tie-breaking** – when all squared distances are 0 we fall back to `Math.random`, sklearn uses the deterministic RandomState and chooses the lowest index.

Both differences cause instability on tiny datasets (≤ 60 samples) used in the fixture suite.

## Acceptance Criteria (the what)

- [ ] Implement sklearn-style potential-based reseeding when a centroid loses all points.
- [ ] Replace calls to `Math.random` with the provided `randomState` stream everywhere inside k-means++.
- [ ] Unit test covering deterministic centroids for a synthetic dataset with duplicate points.
- [ ] No regression in existing K-Means or Spectral tests.

## Implementation Plan (the how)

1. Add helper `choose_furthest_point()` utilising pre-computed distance matrix.
2. Inject `randStream` everywhere (already available via `makeRandomStream`).

