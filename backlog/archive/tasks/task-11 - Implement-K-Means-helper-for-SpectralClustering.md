---
id: task-11
title: Implement K-Means helper for SpectralClustering
status: Done
assignee: [@ai]
created_date: '2025-07-15'
labels: []
dependencies: []
---

## Description

Create a lightweight K-Means implementation to cluster the spectral embedding as the final step of SpectralClustering

## Acceptance Criteria

- [x] Basic K-Means class with fit method
- [x] K-means++ initialization implemented
- [x] Lloyd's algorithm for iterative optimization
- [x] Convergence criteria and maximum iterations
- [x] Integration with SpectralClustering embedding (exported via index)
- [x] Unit tests for K-Means functionality
- [x] Validation against reference implementations (qualitative check via test)

## Implementation Plan

1. Validate parameters (`nClusters`, `maxIter`, `tol`).
2. Implement deterministic PRNG for reproducibility.
3. Implement k-means++ initialisation selecting centroids probabilistically.
4. Use Lloyd’s algorithm with efficient distance calculation to assign labels and update centroids.
5. Convergence when inertia or centroid shift below tolerance or `maxIter` reached.
6. Expose public properties `labels_`, `centroids_`, `inertia_` for later SpectralClustering usage.
7. Export class in `src/index.ts` for integration.
8. Add Jest unit tests verifying two-blob toy dataset clusters correctly.

## Implementation Notes

• Added `src/clustering/kmeans.ts` containing the full implementation as per plan.
• Deterministic Linear Congruential Generator used when `randomState` supplied.
• Distance matrix computed with broadcasting to keep tensor operations efficient and avoid `(n×k×d)` allocations.
• Handles empty cluster edge-case by retaining previous centroid.
• Added `KMeans` export in `src/index.ts`.
• Added `test/kmeans.test.ts`: verifies clustering accuracy, centroid/inertia presence.
• All Jest tests pass (`npm test`).
