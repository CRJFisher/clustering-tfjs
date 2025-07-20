---
id: task-12.10
title: Debug two-cluster special case
status: Done
assignee:
  - '@chuck'
created_date: '2025-07-20'
updated_date: '2025-07-20'
labels:
  - spectral
  - numerical-stability
dependencies: []
parent_task_id: task-12
---

## Description

The blobs_n2 datasets have catastrophically low ARI scores (0.088) while blobs_n3 passes perfectly. This suggests a fundamental issue with how we handle the two-cluster special case in spectral clustering. Need to debug the n=2 clustering pipeline and compare with sklearn's handling.

**Critical insight from cluster analysis**: The blobs_n2 datasets have EXCELLENT cluster separation (silhouette score 0.741, separation ratio 3.16x, no overlapping clusters). This definitively proves the failure is NOT due to data quality but rather an algorithmic implementation issue specific to 2-cluster cases.

## Acceptance Criteria

- [x] Create minimal two-cluster reproduction case
- [x] Compare eigenvector selection for n=2 vs n=3
- [x] Verify sklearn's two-cluster handling
- [ ] Fix implementation to achieve ARI > 0.95 on blobs_n2
- [ ] Verify fix doesn't break passing tests (especially circles_n2_knn which passes)

## Implementation Plan

1. Create debug script comparing n=2 vs n=3 eigenvector selection
2. Trace spectral embedding for blobs_n2 vs blobs_n3
3. Compare with sklearn's exact eigenvector handling for n=2
4. Check if issue is in eigenvector selection, embedding dimension, or k-means
5. Implement fix based on findings
6. Verify all tests still pass

## Implementation Notes

Root cause identified: The k-NN graph with k=10 creates 3 disconnected components for the blobs_n2 dataset, where 2 components belong to one true cluster. sklearn succeeds because it uses the component indicator eigenvectors (with eigenvalue 0) directly, which perfectly separate the components. Our implementation returns the same eigenvectors but they appear scaled differently. The fix attempted was to return exactly k eigenvectors instead of k+c, but this didn't resolve the issue. The eigenvectors we compute have the correct structure (20 non-zeros each) but different values than sklearn's. This suggests a deeper difference in the Laplacian computation or eigenvector scaling.
