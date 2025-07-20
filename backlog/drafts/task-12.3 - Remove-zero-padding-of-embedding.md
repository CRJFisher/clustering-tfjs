---
id: task-12.3
title: Remove zero-padding when fewer than k informative eigenvectors exist
status: To Do
assignee: []
created_date: '2025-07-18'
labels: [spectral]
dependencies: [task-12.1]
---

## Description (the why)

When the Laplacian yields fewer than `nClusters` non-trivial eigenvectors we currently append columns of **zeros** so that K-Means can still run.  scikit-learn instead raises `ValueError("n_components must be < number of samples in the graph")` which prevents the later steps from producing degenerate embeddings.

Zero-padding destroys pairwise distances (all added dimensions are identical) and is one of the reasons why several ARI fixtures fail.

## Acceptance Criteria (the what)

- [ ] If after discarding trivial vectors fewer than `nClusters` columns remain, `SpectralClustering.fit()` throws a clear error.
- [ ] Unit test: requesting 3 clusters from a line graph of 2 nodes raises.
- [ ] Zero-padding code path removed and associated test updated/removed.

## Implementation Plan (the how)

1. Delete padding branch in `spectral.ts`.
2. Update downstream tests that expected silent zero-padding.

