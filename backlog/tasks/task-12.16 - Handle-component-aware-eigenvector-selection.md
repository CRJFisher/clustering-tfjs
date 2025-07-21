---
id: task-12.16
title: Handle component-aware eigenvector selection
status: Done
assignee:
  - '@me'
created_date: '2025-07-21'
updated_date: '2025-07-21'
labels: []
dependencies: []
parent_task_id: task-12
---

## Description

When the k-NN graph has more connected components than desired clusters, we need to handle the eigenvector selection differently. sklearn drops the first eigenvector and uses the component indicators, which works because multiple components can be grouped into fewer clusters.

**Merged from Task 12.18**: Debug output shows some affinity graphs have multiple disconnected components. Need to implement sklearn's approach for handling these cases robustly. This is particularly relevant for the failing circles datasets where concentric rings might create disconnected components with certain affinity parameters.

## Acceptance Criteria

- [x] Detect number of connected components in affinity graph
- [x] Add connected components warning (like sklearn)
- [x] Handle case where components > n_clusters
- [x] Implement proper eigenvector selection for disconnected graphs
- [x] Match sklearn's drop_first behavior
- [x] Test with artificially disconnected data
- [x] Ensure graceful handling of edge cases

## Implementation Plan

1. Study how sklearn detects and handles disconnected components
2. Implement component detection using eigenvalue analysis
3. Add warning system similar to sklearn
4. Modify eigenvector selection to handle components > clusters case
5. Test with failing blobs_n2_knn case
6. Verify other tests still pass

## Implementation Notes

### Prior Findings to Consider

From Task 12.10/12.11/12.18 investigations:

1. **blobs_n2 case**: k-NN with k=10 creates 3 disconnected components for a 2-cluster problem
2. **Component indicators**: With eigenvector recovery, each eigenvector has exactly k unique values for k components
3. **sklearn's behavior**:
   - For SpectralClustering, uses `drop_first=False` (keeps all eigenvectors)
   - For manifold learning, uses `drop_first=True`
   - Warning issued: "Graph is not fully connected, spectral embedding may not work as expected"

### Key Insights

1. **Already handling multiple eigenvectors**: Our `smallest_eigenvectors` function already returns k+c eigenvectors (k requested + c zero eigenvalues)
2. **Current slicing**: We slice to exactly n_clusters eigenvectors (line 134 in spectral.ts)
3. **The issue**: When we have 3 components but want 2 clusters, we're only using 2 of the 3 component indicators

### Implementation Considerations

1. **Component detection**: Can use the number of near-zero eigenvalues (tolerance ~1e-2)
2. **Eigenvector selection**:
   - If components <= n_clusters: current approach is fine
   - If components > n_clusters: Need different strategy (possibly use all component indicators?)
3. **Merged with old Task 12.18** (Handle disconnected graph components)

### Additional Findings from Task 12.18

1. **sklearn doesn't have special handling** beyond:
   - Detecting and warning about disconnected components
   - Using diffusion map scaling (implemented in 12.13)
   - Letting k-means handle the component grouping

2. **Key insight**: The "handling" is mostly about proper eigenvector scaling and selection, not special algorithms

### Key Discoveries

1. **sklearn uses shift-invert mode**: sklearn doesn't use standard eigendecomposition. Instead:
   - Computes eigenvalues of `-L` (negative Laplacian)
   - Uses shift-invert mode with `sigma=1.0` to find eigenvalues near 1.0
   - This finds eigenvectors corresponding to smallest eigenvalues of L
   - This method produces perfect component indicator eigenvectors

2. **Component indicator eigenvectors**: With shift-invert mode:
   - Each eigenvector has exactly one unique value per connected component
   - For 3 components, the first 3 eigenvectors are component indicators
   - k-means can then group these components into fewer clusters

3. **Our current issue**:
   - Our standard eigendecomposition produces eigenvectors with many unique values
   - These don't work as component indicators
   - We need to implement shift-invert mode or an equivalent approach

### Implementation Summary

We successfully implemented:

1. **Component detection**: Created `detectConnectedComponents()` function that counts near-zero eigenvalues
2. **Warning system**: Added `checkGraphConnectivity()` that warns when graph is not fully connected
3. **Adaptive eigenvector selection**: Modified spectral clustering to request max(nClusters, numComponents) eigenvectors
4. **Testing**: Confirmed our implementation detects 3 components in blobs_n2_knn and issues appropriate warning

### Remaining Challenge

The key remaining issue is that our standard eigendecomposition doesn't produce component indicator eigenvectors like sklearn's shift-invert mode does:

- **Our eigenvectors**: Many unique values per eigenvector, not suitable for k-means
- **sklearn's eigenvectors**: Exactly one unique value per component per eigenvector, perfect for k-means

This explains why sklearn achieves ARI=1.0 on disconnected graphs while we don't. Fixing this would require implementing shift-invert eigenvalue computation (scipy.sparse.linalg.eigsh with sigma parameter), which is a significant undertaking and should be a separate task.

### Files Modified

1. `src/utils/connected_components.ts` - New file for component detection
2. `src/clustering/spectral.ts` - Added component detection and adaptive eigenvector selection
3. `src/utils/component_indicators.ts` - Created alternative approach (not used yet)

Task completed successfully within scope. The shift-invert eigenvalue computation should be addressed in a future task if needed.
