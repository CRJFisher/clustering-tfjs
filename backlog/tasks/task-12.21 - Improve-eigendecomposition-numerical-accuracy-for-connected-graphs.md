---
id: task-12.21
title: Improve eigendecomposition numerical accuracy for connected graphs
status: In Progress
assignee:
  - '@me'
created_date: '2025-07-21'
updated_date: '2025-07-21'
labels: []
dependencies: []
parent_task_id: task-12
---

## Description

The 5 remaining failing tests (circles_n2_rbf, circles_n3_knn, circles_n3_rbf, moons_n2_rbf, moons_n3_rbf) are all connected graphs where sklearn achieves ARI=1.0 using standard eigendecomposition. Our implementation produces different eigenvectors, causing k-means to fail. Need to improve numerical accuracy to match sklearn's ARPACK results.

## Acceptance Criteria

- [ ] Eigendecomposition produces eigenvectors matching sklearn's quality
- [ ] All 5 remaining connected graph tests pass
- [ ] No regression in existing passing tests

## Implementation Plan

1. Analyze current eigendecomposition implementation (improved Jacobi)
2. Create detailed comparison of our vs sklearn eigenvectors on failing tests
3. Identify specific numerical differences (eigenvalue accuracy, eigenvector alignment)
4. Test alternative approaches: increase Jacobi iterations/tolerance, try different algorithms
5. Consider using ARPACK bindings or other high-precision solvers
6. Implement the most effective solution
7. Verify all 5 remaining tests pass

## Implementation Notes

After extensive investigation, found that:

1. Our eigenvector computation IS slightly different from sklearn's ARPACK
2. The first element matches after scaling, but others differ by up to 0.0065
3. This small difference is enough to reduce ARI from 1.0 to 0.8689
4. The issue is NOT about constant eigenvector replacement or scaling
5. We need to either: (a) use ARPACK like sklearn, or (b) improve our Jacobi solver accuracy

After extensive investigation, found that:

1. Our eigenvector computation IS slightly different from sklearn's ARPACK
2. The first element matches after scaling, but others differ by up to 0.0065
3. This small difference is enough to reduce ARI from 1.0 to 0.8689
4. The issue is NOT about constant eigenvector replacement or scaling
5. We need to either: (a) use ARPACK like sklearn, or (b) improve our Jacobi solver accuracy

## Detailed Investigation Results

### Initial Hypothesis: Constant Eigenvector Issue

- sklearn replaces the first eigenvector with a theoretical constant for connected graphs
- Implemented this approach but ARI remained at 0.8689
- Found that using just dimension 1 (second eigenvector) gives perfect ARI in sklearn
- This ruled out the constant eigenvector as the issue

### Root Cause: Eigenvector Computation Differences

- Compared our eigenvectors with sklearn's element by element
- First element matches perfectly after scaling: -0.0375329558
- But other elements differ:
  - Element 1: sklearn=0.0061266530, ours=0.0062234214 (diff=-0.0000967684)
  - Element 2: sklearn=0.0069854324, ours=0.0087891251 (diff=-0.0018036926)
  - Up to 0.0065 difference in some elements
- These small differences cause k-means to cluster differently

### Why This Matters

- Even tiny eigenvector differences can flip cluster assignments
- K-means is sensitive to initial conditions and small perturbations
- sklearn achieves ARI=1.0, we achieve ARI=0.8689 with these small differences

### Current Implementation

- Using improved Jacobi method with tolerance=1e-14, maxIterations=3000
- This is already very tight, but Jacobi has inherent limitations vs ARPACK
- ARPACK (used by sklearn) is specifically designed for finding extreme eigenvalues

### Accuracy Tests

Tested higher accuracy settings:

- Current: 3000 iterations, 1e-14 tolerance
- Higher: 10000 iterations, 1e-16 tolerance
- Result: No improvement - eigenvectors identical
- Conclusion: We've hit the limit of Jacobi method accuracy

### Implementation Options Evaluated

#### Option 1: Accept Current Accuracy

- We pass 7/12 tests (58%)
- Document that small numerical differences exist vs sklearn
- Note that ARI ~0.87 is still good clustering

#### Option 2: Native Node.js Addon with ARPACK

Investigated two approaches:

**2a. Using Spectra (C++ ARPACK reimplementation)**

- Pros:
  - Modern C++ header-only library
  - Clean API designed for sparse eigenvalues
  - No Fortran dependencies
  - Good documentation
- Cons:
  - Need Node-API/N-API bindings
  - Platform-specific builds needed
  - Distribution complexity
- Difficulty: Medium-High

**2b. Direct ARPACK bindings**

- Pros:
  - Exact match with sklearn
- Cons:
  - ARPACK is Fortran
  - Very complex build setup
  - Platform dependencies
- Difficulty: Very High

#### Option 3: WebAssembly Compilation

Investigated compiling ARPACK to WASM:

**3a. F2C Approach (proven)**

- Convert Fortran → C → WASM
- Someone created arpack-js using this
- Complex but works

**3b. Direct Fortran → WASM**

- LFortran supports it (experimental)
- LLVM Flang needs patches
- 32-bit WASM vs 64-bit ARPACK issues

**Pros:**

- Platform independent
- Bundle with npm easily
- No native dependencies

**Cons:**

- Complex build process
- Performance overhead
- Experimental toolchain

**Difficulty: High**

### Recommendation

1. First try to make k-means less sensitive to small eigenvector differences
2. If that fails, implement Spectra native addon (most practical)
3. WebAssembly only if platform independence is critical

The eigenvector differences are real but small - the issue may be more about k-means sensitivity than eigendecomposition accuracy.

## Detailed Investigation Results

### Initial Hypothesis: Constant Eigenvector Issue
