---
id: task-12.13
title: Implement eigenvector recovery for normalized Laplacian
status: To Do
assignee: []
created_date: '2025-07-21'
labels: []
dependencies: []
parent_task_id: task-12
---

## Description

Add the missing eigenvector recovery step that sklearn performs. After computing eigenvectors of the normalized Laplacian L = I - D^(-1/2) * A * D^(-1/2), we need to divide by sqrt(degree) to recover the true eigenvectors. This transforms them to have k unique values for k components.

## Acceptance Criteria

- [ ] Add degree computation in normalised_laplacian function
- [ ] Implement eigenvector recovery in SpectralClustering.fit()
- [ ] Update the incorrect comment about raw eigenvectors
- [ ] Verify eigenvectors have correct number of unique values

## Implementation Notes

### Prior Findings to Consider

From Task 12.12/12.18 investigation:
1. **Mathematical formula**: Recovery is `u = v / sqrt(degree)` where v is the eigenvector
2. **Test results confirm**: With recovery, eigenvectors have exactly k unique values for k components
3. **Location of incorrect comment**: Lines 132-133 in spectral.ts claim sklearn uses "raw eigenvectors WITHOUT any additional scaling"
4. **Important**: Even with recovery, ARI remains 0.0876 (sklearn gets 1.0), so this alone won't fix all tests

### Technical Details

The recovery transforms eigenvectors from the normalized Laplacian to eigenvectors of the random walk Laplacian, which naturally encode component membership.

### Where to Implement

1. In `normalised_laplacian()`: Return both the Laplacian and the degree vector (or sqrt of degrees)
2. In `SpectralClustering.fit()`: After getting eigenvectors, divide by sqrt(degrees)
3. Note: sklearn uses `csgraph.laplacian(affinity, normed=True, return_diag=True)` which returns both
