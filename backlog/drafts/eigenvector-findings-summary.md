# Eigenvector Computation Findings Summary

This document compiles all findings related to eigenvectors, eigenvalues, and eigenvector computation from the SpectralClustering implementation effort (Task 12 and its subtasks).

## CRITICAL UPDATE (Task 12.13): sklearn Uses Diffusion Map Scaling, NOT Eigenvector Recovery!

After extensive investigation in Task 12.13, we discovered that our initial understanding was incorrect:

### What sklearn ACTUALLY does:

1. Computes eigenvectors of normalized Laplacian: `L = I - D^(-1/2) * A * D^(-1/2)`
2. **Applies diffusion map scaling**: scales eigenvectors by `sqrt(1 - eigenvalue)`
3. This is the standard diffusion map embedding for normalized Laplacian
4. **Does NOT use eigenvector recovery** (dividing by sqrt(degrees))

### The Confusion:

- **Initial belief**: sklearn divides by sqrt(degrees) for eigenvector recovery
- **Reality**: sklearn scales by sqrt(1 - eigenvalue) for diffusion maps
- **Why it matters**: Eigenvector recovery helps disconnected graphs but hurts connected ones

### Impact of Diffusion Map Scaling:

- Fixed all "blobs" datasets (disconnected components) - ARI = 1.0
- Improved but didn't fix "moons" and "circles" datasets - ARI > 0.93 but < 0.95
- k-NN graphs remain problematic due to graph structure issues

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

### Updated Understanding (Task 12.13):

1. **sklearn uses diffusion map scaling**: `eigenvectors * sqrt(1 - eigenvalue)`
2. **NOT eigenvector recovery**: Does not divide by sqrt(degrees)
3. **Our initial comment was actually correct**: sklearn does use "raw eigenvectors" (plus diffusion scaling)

### What We Implemented:

1. **Created `smallest_eigenvectors_with_values`** to return both eigenvectors and eigenvalues
2. **Implemented diffusion map scaling** in SpectralClustering.fit()
3. **Removed incorrect eigenvector recovery** that was hurting connected graphs

### Remaining Issues:

1. **k-NN graphs produce non-constant eigenvectors**: The constant vector 1/sqrt(n) is NOT an eigenvector
2. **sklearn somehow gets constant first embedding**: Suggesting additional processing or different eigensolver
3. **Several tests are close but not perfect**: ARI > 0.93 but < 0.95

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

## Conclusions (Updated after Task 12.13)

1. **Eigensolver is mostly fine**: Jacobi solver works correctly for most cases
2. **Diffusion map scaling is essential**: Must scale by sqrt(1 - eigenvalue), NOT eigenvector recovery
3. **k-NN graphs have structural issues**: The graph connectivity prevents ideal spectral embeddings
4. **Not a precision issue**: float32 is sufficient; problems are algorithmic
5. **Progress made**: Fixed all disconnected component cases (blobs), improved connected cases (moons/circles)

## Key Lesson from Task 12.13

The journey from eigenvector recovery to diffusion map scaling illustrates the importance of:
- Testing hypotheses thoroughly before implementation
- Understanding that different graph types (connected vs disconnected) may need different approaches
- Recognizing that sklearn's implementation has subtleties not apparent from documentation alone

The diffusion map scaling is the correct foundation, but perfect sklearn parity requires addressing the k-NN graph structure issues that remain.

## Next Steps

1. ~~Implement eigenvector recovery (Task 12.13)~~ ✓ Implemented diffusion map scaling instead
2. Investigate why sklearn gets better embeddings for k-NN graphs (Task 12.14+)
3. Consider alternative eigensolvers or preprocessing for k-NN cases
4. Explore component-aware strategies when components > clusters

_Document compiled from Tasks 12.1-12.21 findings on 2025-07-21_
