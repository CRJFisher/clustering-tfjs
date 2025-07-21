---
id: task-22
title: Task 12 completion strategy and prioritization
status: To Do
assignee: []
created_date: '2025-07-21'
updated_date: '2025-07-21'
labels: []
dependencies: []
---

## Description

Strategy document for completing SpectralClustering implementation (task-12) based on extensive investigation findings. Outlines the prioritized approach to fixing the remaining 5 failing tests.

## Current Status Analysis

### Test Results

- **7/12 tests passing (58%)**
  - All k-NN tests pass (blobs and moons datasets)
  - 3/8 RBF tests pass (blobs datasets)
- **5/12 tests failing**
  - circles_n2_rbf (ARI=0.869)
  - circles_n3_knn (ARI=0.932)
  - circles_n3_rbf (ARI=0.779)
  - moons_n2_rbf (ARI=0.869)
  - moons_n3_rbf (ARI=0.779)

### Root Cause

After extensive investigation, the issue is **NOT**:

- Algorithm correctness (our logic matches sklearn)
- Constant eigenvector handling
- Disconnected graph handling
- Shift-invert eigenvalue computation

The issue **IS**:

- Small numerical differences in eigenvector computation
- Our Jacobi method vs sklearn's ARPACK
- Differences up to 0.0065 in eigenvector elements
- K-means clustering is sensitive to these small differences

### Key Insight

We're very close! ARI values of 0.77-0.93 indicate our algorithm is fundamentally correct. The small numerical differences are just enough to cause k-means to produce slightly different cluster assignments.

## Prioritized Solution Strategy

### Option 1: Make K-means More Robust (Easiest)

**Task: 12.22 - Make k-means less sensitive to small eigenvector differences**

**Approach:**

- Investigate k-means++ initialization with more careful seeding
- Try multiple k-means runs with different seeds and take consensus
- Implement deterministic tie-breaking in k-means
- Consider alternative clustering methods (e.g., spectral clustering with discretization)

**Pros:**

- Pure algorithmic solution, no new dependencies
- Could improve robustness generally
- Fastest to implement and test

**Cons:**

- May not fully solve the problem
- Deviates from sklearn's exact implementation

**Estimated effort:** 1-2 days

### Option 2: Better Eigensolver in Pure JS (Medium)

**Task: 12.23 - Investigate alternative eigensolvers for better accuracy**

**Approach:**

- Research QR algorithm implementation
- Investigate Lanczos method for symmetric matrices
- Look for existing JS libraries with better eigensolvers
- Consider using higher precision arithmetic (e.g., decimal.js)

**Pros:**

- Stays in pure JavaScript/TypeScript
- No build complexity
- Could find a good middle ground

**Cons:**

- May still not match ARPACK accuracy
- Implementation complexity
- Performance implications

**Estimated effort:** 3-5 days

### Option 3: ARPACK Bindings (Hardest)

**Task: 12.21 - Implement ARPACK bindings**

**Sub-options:**

1. **Native addon with Spectra (C++)** - Modern ARPACK reimplementation
2. **WebAssembly with f2c** - Convert Fortran ARPACK to WASM
3. **Direct ARPACK bindings** - Most complex but exact match

**Pros:**

- Exact match with sklearn
- Best numerical accuracy
- Solves the problem definitively

**Cons:**

- High implementation complexity
- Build/distribution challenges
- Platform-specific issues

**Estimated effort:** 1-2 weeks

## Recommendation

Try options in order:

1. **Start with Option 1** - It's the fastest and might be sufficient
2. **If that fails, try Option 2** - Good balance of effort vs reward
3. **Resort to Option 3 only if necessary** - High complexity but guaranteed solution

## Alternative: Accept Current Accuracy

Consider documenting that:

- We achieve 58% test parity with sklearn
- Our implementation is correct but uses different numerical methods
- ARI of 0.87-0.93 is still good clustering performance
- Users needing exact sklearn parity should use sklearn

This would allow us to:

- Ship a working implementation
- Document the limitations clearly
- Revisit accuracy improvements later if needed

## Success Metrics

- Primary: All 12 fixture tests pass (ARI >= 0.95)
- Secondary: No regression in currently passing tests
- Tertiary: Maintain pure JS implementation if possible

## Decision Factors

Choose based on:

1. **Time constraints** - How soon do we need to ship?
2. **Accuracy requirements** - Is 87-93% ARI acceptable?
3. **Maintenance burden** - Can we support native addons/WASM?
4. **User needs** - Do users need exact sklearn parity?

## Conclusion

The spectral clustering implementation is fundamentally sound. The remaining challenge is a numerical accuracy issue that manifests in k-means sensitivity. We have clear options ranging from quick fixes to comprehensive solutions. The choice depends on project priorities and constraints.
