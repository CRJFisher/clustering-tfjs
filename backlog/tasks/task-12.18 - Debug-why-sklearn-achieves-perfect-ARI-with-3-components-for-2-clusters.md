---
id: task-12.18
title: Debug why sklearn achieves perfect ARI with 3 components for 2 clusters
status: To Do
assignee: []
created_date: '2025-07-21'
labels: []
dependencies: []
parent_task_id: task-12
---

## Description

Even with correct eigenvectors (3 unique values each), we still get ARI=0.0876 while sklearn achieves ARI=1.0. Need to trace through sklearn's exact process to understand how it handles the case where number of components > number of clusters.

## Acceptance Criteria

- [ ] Trace sklearn's full spectral clustering pipeline
- [ ] Identify how sklearn handles component-cluster mismatch
- [ ] Determine if sklearn uses special k-means initialization
- [ ] Test if the issue is in eigenvector selection or k-means

## Implementation Notes

### Prior Findings to Consider

From Task 12.12/12.18 investigation:

1. **Confirmed**: Even with eigenvector recovery (giving 3 unique values), ARI = 0.0876
2. **sklearn achieves ARI = 1.0** with the same data and recovered eigenvectors
3. **Key insight**: The embedding has 3 unique values per dimension, effectively encoding which of the 3 components each point belongs to
4. **The mystery**: How does sklearn's k-means cluster 3 components into 2 clusters perfectly?

### Hypotheses to Test

1. **Eigenvector selection**: Does sklearn select different eigenvectors when components > clusters?
2. **drop_first behavior**: sklearn uses `drop_first=True` in spectral_embedding - is this different for SpectralClustering?
3. **K-means initialization**: Does sklearn use a special initialization for this case?
4. **Component merging**: Does sklearn have logic to merge components when components > clusters?

### Key Code Locations

From previous investigation:

- sklearn calls `_spectral_embedding` with `drop_first=False` for SpectralClustering (line 754 in spectral.py)
- This differs from manifold learning which uses `drop_first=True`
