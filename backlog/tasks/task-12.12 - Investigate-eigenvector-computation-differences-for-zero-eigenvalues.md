---
id: task-12.12
title: Investigate eigenvector computation differences for zero eigenvalues
status: Done
assignee:
  - '@claude'
created_date: '2025-07-21'
updated_date: '2025-07-21'
labels: []
dependencies: []
parent_task_id: task-12
---

## Description

Our Jacobi solver produces different eigenvectors than scipy's ARPACK for the degenerate case of multiple zero eigenvalues. When there are 3 connected components (3 zero eigenvalues), sklearn's eigenvectors have exactly 3 unique values each (component indicators), while ours have 14-15 unique values.

## Acceptance Criteria

- [x] Understand why Jacobi produces different eigenvectors
- [x] Document the mathematical reason for the difference
- [x] Determine if we need to switch eigensolvers or post-process

## Implementation Notes

### Root Cause Identified

The issue is NOT with the eigensolver itself, but with a missing post-processing step called **eigenvector recovery**.

### Key Findings

1. **Both Jacobi and ARPACK find the same eigenspace**: The 3D null space (for 3 components) is identical, but they choose different orthonormal bases within this space.

2. **The real issue**: We're missing the eigenvector recovery step that sklearn performs for normalized Laplacian eigenvectors.

3. **What sklearn does**:
   - Computes eigenvectors of the normalized Laplacian: `L = I - D^(-1/2) * A * D^(-1/2)`
   - Recovers the eigenvectors by dividing by `D^(1/2)`: `u = v / sqrt(degree)`
   - This transforms the eigenvectors to have exactly k unique values (one per component)

4. **Test results**:
   - Without recovery: 39 unique values per eigenvector, ARI = 0.0876
   - With recovery: 3 unique values per eigenvector (matching sklearn), but ARI still = 0.0876
5. **Why ARI is still low**: The eigenvector recovery alone doesn't fix the clustering because:
   - We have 3 components but only 2 desired clusters
   - sklearn achieves ARI = 1.0 using additional logic we haven't identified yet
   - The comment in our code claiming sklearn uses "raw eigenvectors" is incorrect

### Mathematical Explanation

For a normalized Laplacian with disconnected components:

- The eigenvectors of `L = I - D^(-1/2) * A * D^(-1/2)` are not component indicators
- But the recovered eigenvectors `u = v / sqrt(degree)` ARE component indicators
- This is because we're essentially computing eigenvectors of the random walk Laplacian

### Recommendation

1. **Immediate fix**: Implement eigenvector recovery in `SpectralClustering.fit()`
2. **Further investigation**: Determine why sklearn achieves ARI = 1.0 while we get 0.0876 even with correct eigenvectors
3. **No need to change eigensolvers**: Jacobi works fine; the issue was post-processing
