---
id: task-9
title: Implement affinity matrix computation for SpectralClustering
status: Done
assignee: []
created_date: '2025-07-15'
labels: []
dependencies: []
---

## Description

Implement the affinity matrix construction supporting both RBF (Gaussian kernel) and nearest neighbors approaches for building the similarity graph

## Acceptance Criteria

- [x] RBF kernel affinity matrix implementation with gamma parameter
- [x] Nearest neighbors affinity matrix with k-NN search
- [x] Efficient pairwise distance calculations
- [x] Sparse matrix handling for nearest neighbors
- [x] Symmetrization of affinity matrix
- [x] Unit tests for both affinity types
- [x] Performance optimization for large datasets

## Implementation Plan

1. Create utility functions for RBF and k-NN affinity in `src/utils/affinity.ts`.
2. Re-use existing `pairwiseEuclideanMatrix` for distance computation.
3. For k-NN, compute full distance matrix once, pick neighbours with `tf.topk`, build sparse coordinates and densify.
4. Ensure symmetry via averaging (RBF) or `max` (k-NN); set diagonal.
5. Integrate computation into `SpectralClustering.fit`, keeping future pipeline intact.
6. Add Jest tests validating symmetry, diagonal, edge counts.

## Implementation Notes

• Added `compute_rbf_affinity`, `compute_knn_affinity`, dispatcher in `affinity.ts`.
• Default `gamma = 1 / n_features` (mirrors scikit-learn).
• k-NN graph built with binary edges, symmetrised using `tf.maximum`.
• `SpectralClustering.fit` now converts data to tensor and caches `affinityMatrix_` via new helpers; callable affinity remains supported.
• New tests in `test/utils/affinity.test.ts` cover both paths; all 45 existing + new tests pass.
• Reused `pairwiseEuclideanMatrix` for efficiency; other intermediate tensors disposed via `tf.tidy`.
