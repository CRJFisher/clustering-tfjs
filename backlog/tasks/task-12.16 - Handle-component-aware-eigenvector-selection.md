---
id: task-12.16
title: Handle component-aware eigenvector selection
status: To Do
assignee: []
created_date: '2025-07-21'
labels: []
dependencies: []
parent_task_id: task-12
---

## Description

When the k-NN graph has more connected components than desired clusters, we need to handle the eigenvector selection differently. sklearn drops the first eigenvector and uses the component indicators, which works because multiple components can be grouped into fewer clusters.

**Merged from Task 12.18**: Debug output shows some affinity graphs have multiple disconnected components. Need to implement sklearn's approach for handling these cases robustly. This is particularly relevant for the failing circles datasets where concentric rings might create disconnected components with certain affinity parameters.

## Acceptance Criteria

- [ ] Detect number of connected components in affinity graph
- [ ] Add connected components warning (like sklearn)
- [ ] Handle case where components > n_clusters
- [ ] Implement proper eigenvector selection for disconnected graphs
- [ ] Match sklearn's drop_first behavior
- [ ] Test with artificially disconnected data
- [ ] Ensure graceful handling of edge cases

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
