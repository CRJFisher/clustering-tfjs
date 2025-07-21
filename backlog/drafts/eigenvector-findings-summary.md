# Eigenvector Computation Findings Summary

This document compiles all findings related to eigenvectors, eigenvalues, and eigenvector computation from the SpectralClustering implementation effort (Task 12 and its subtasks).

## Critical Discovery: Eigenvector Recovery

The most important finding is that we're missing a critical post-processing step called **eigenvector recovery** that sklearn performs for normalized Laplacian eigenvectors.

### What sklearn does:

1. Computes eigenvectors of normalized Laplacian: `L = I - D^(-1/2) * A * D^(-1/2)`
2. **Recovers eigenvectors** by dividing by `D^(1/2)`: `u = v / sqrt(degree)`
3. This transforms eigenvectors to have exactly k unique values for k connected components

### Impact:

- **Without recovery**: Eigenvectors have many unique values (e.g., 39 for blobs_n2)
- **With recovery**: Eigenvectors have exactly k unique values (e.g., 3 for 3 components)
- This is critical for handling disconnected or weakly connected graphs

## Timeline of Discoveries

### Early Findings (Tasks 12.1-12.6)

1. **Eigenpair Determinism (Task 12.3)**
   - Problem: Eigensolvers produce non-deterministic results due to ordering and sign ambiguity
   - Solution: Implemented deterministic eigenpair processing (sort by eigenvalue, consistent sign flipping)
   - Result: Reproducible eigenvector computations

2. **Jacobi Solver Accuracy (Task 12.4.1)**
   - Problem: Jacobi solver produced negative eigenvalues for PSD matrices
   - Solution: Improved solver with cyclic sweeps, adaptive thresholds, and PSD-specific handling
   - Result: Non-negative eigenvalues (clamping threshold: 1e-8)

3. **Scaling Misconception (Task 12.6)**
   - Initial belief: sklearn uses 'diffusion map' scaling (multiply by sqrt(1-eigenvalue))
   - Truth: sklearn uses raw eigenvectors without additional scaling
   - Lesson: The D^(-1/2) normalization is already in the eigenvectors

### Precision Investigation (Task 12.8)

- Finding: TensorFlow.js only supports float32, not float64
- Conclusion: Precision limitations are NOT causing the failures
- Implication: Issues are algorithmic, not numerical precision

### Disconnected Components Discovery (Tasks 12.10-12.11)

1. **Multiple Components (Task 12.10)**
   - Discovery: k-NN graph with k=10 creates 3 disconnected components for blobs_n2 dataset
   - sklearn's approach: Uses component indicator eigenvectors (with eigenvalue 0)
   - Our issue: Correct structure but wrong numerical values

2. **Unique Values Pattern (Task 12.11)**
   - sklearn's eigenvectors: Exactly 3 unique values per column (component indicators)
   - Our eigenvectors: 14-15 unique values per column
   - Root cause: Different eigenvector basis for degenerate eigenspace

### Final Breakthrough (Task 12.12/12.18)

1. **Eigenspace Analysis**
   - Both Jacobi and ARPACK find the same 3D null space
   - They choose different orthonormal bases within this space
   - The issue is NOT the eigensolver but the missing recovery step

2. **Eigenvector Recovery**
   - Mathematical reason: Eigenvectors of normalized Laplacian need transformation
   - The recovery `u = v / sqrt(degree)` converts to random walk Laplacian eigenvectors
   - This produces component indicator vectors with k unique values

3. **Remaining Mystery**
   - Even with correct eigenvectors (3 unique values), ARI = 0.0876
   - sklearn achieves ARI = 1.0 with the same data
   - Additional logic needed for handling components > clusters case

## Key Implementation Issues

### Current Problems:

1. **Missing eigenvector recovery** in `SpectralClustering.fit()`
2. **Incorrect comment** claiming sklearn uses "raw eigenvectors" (line 132-133 in spectral.ts)
3. **No component-aware logic** for when #components > #clusters

### Required Fixes:

1. **Immediate (Task 12.13)**:
   - Add degree vector computation to normalized Laplacian
   - Implement eigenvector recovery: `embedding = eigenvectors / sqrt(degrees)`
   - Update misleading comments

2. **Investigation (Task 12.14)**:
   - Trace sklearn's exact handling of 3 components → 2 clusters
   - Understand why recovery alone doesn't achieve ARI = 1.0

3. **Component Handling (Task 12.15)**:
   - Implement logic for components > clusters
   - Consider sklearn's `drop_first` approach

## Technical Details

### Eigenvector Properties by Method:

| Method           | Unique Values | Value Range Ratio | ARI Score |
| ---------------- | ------------- | ----------------- | --------- |
| Raw Jacobi       | 39            | 76-492x           | 0.0876    |
| Raw ARPACK       | 39            | 3-15x             | 0.0876    |
| Recovered Jacobi | 3             | 44-279x           | 0.0876    |
| sklearn          | 3             | ~1x               | 1.000     |

### Zero Eigenvalue Handling:

- For k disconnected components: k zero eigenvalues
- Eigenvectors span k-dimensional null space
- ANY orthonormal basis is mathematically valid
- Component indicators are the "natural" choice

## Conclusions

1. **Eigensolver is fine**: Jacobi solver works correctly; no need to switch to ARPACK
2. **Recovery is essential**: Must implement eigenvector recovery for normalized Laplacian
3. **Components need special handling**: When #components > #clusters, additional logic required
4. **Not a precision issue**: float32 is sufficient; problems are algorithmic

## Next Steps

1. Implement eigenvector recovery (Task 12.13)
2. Debug why sklearn achieves perfect clustering (Task 12.14)
3. Add component-aware eigenvector selection (Task 12.15)

---

_Document compiled from Tasks 12.1-12.21 findings on 2025-07-21_
