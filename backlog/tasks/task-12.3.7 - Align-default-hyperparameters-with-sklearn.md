---
id: task-12.3.7
title: Align default hyper-parameters with scikit-learn
status: To Do
assignee: []
created_date: '2025-07-17'
labels: []
dependencies: [task-12.3.6]
---

## Description (the why)

Spectral-clustering fixtures generated with scikit-learn rely on its *implicit* defaults:

• RBF affinity → `gamma = 1 / n_features`  
• k-NN affinity → `n_neighbors = round(log2(n_samples))`

Our implementation currently hard-codes `gamma = 1.0` and `nNeighbors = 10` whenever the user does not supply an explicit value.  This discrepancy is the primary cause of the remaining ARI mismatch for the **k-NN** fixtures and was previously responsible for the RBF failures fixed in task-12.3.5.

## Acceptance Criteria (the what)

- [ ] `compute_rbf_affinity` already uses the corrected γ default; add unit test guarding against regression.
- [ ] `SpectralClustering.defaultNeighbors()` changed to:
      ```ts
      const n = params.nNeighbors ?? Math.round(Math.log2(nSamples));
      ```
      where `nSamples` is available at runtime of `fit()`; if unavailable fall back to 10.
- [ ] When `params.nNeighbors` is provided it must *always* override the heuristic.
- [ ] Update JSDoc explaining the scikit-learn alignment.
- [ ] Add Jest tests asserting:
    1. For a 256-sample dataset the default k = 8 (log2).
    2. Passing `nNeighbors` overrides the heuristic.
- [ ] All **knn** fixtures in `test/fixtures/spectral/` reach ARI ≥ 0.95 (float32 pipeline).

## Implementation Plan (the how)

1. In `SpectralClustering.fit()` determine `effectiveK` early (`params.nNeighbors ?? heuristic`).  Pass it to `compute_knn_affinity`.
2. Write small utility test dataset (circle blobs) to validate heuristic.
3. Adjust docs & CHANGELOG.

## Dependencies

Must follow tasks 12.3.5 & 12.3.6; independent of 12.3.8.

## Implementation Notes

*To be filled after completion*
