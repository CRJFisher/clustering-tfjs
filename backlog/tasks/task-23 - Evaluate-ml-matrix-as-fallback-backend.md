---
id: task-23
title: Evaluate ml-matrix as fallback backend
status: To Do
assignee: []
created_date: '2025-07-22'
updated_date: '2025-07-22'
labels: []
dependencies:
  - task-19
  - task-19.1
---

## Description

Evaluate whether to keep ml-matrix as a fallback linear algebra backend when TensorFlow.js backends are unavailable or underperforming. Eigendecomposition in ml-matrix for SpectralClustering is 10x faster than our Jacobi implementation. We should also evaluate broader usage of ml-matrix in other places.

## Acceptance Criteria

- [ ] Benchmark ml-matrix vs TensorFlow.js for key operations:
  - [ ] Matrix multiplication
  - [ ] Eigendecomposition (already proven 10x faster)
  - [ ] Distance calculations
  - [ ] Other linear algebra operations
- [ ] Identify scenarios where ml-matrix is beneficial:
  - [ ] When tfjs-node fails to install
  - [ ] In restricted environments (some corporate networks)
  - [ ] For specific operations where it's consistently faster
- [ ] Evaluate package size impact
- [ ] Decision: Keep ml-matrix, remove it, or make it optional
- [ ] Also evaluate `eigen.js`'s `Solvers.eigenSolve` for eigendecomposition. Check the ARI and numerical stability of fixture tests SpectralClustering with ml-matrix and eigen.js. `eigen.js` also has `Decompositions` for matrix decompositions algorithms.

## Implementation Plan

### Key Considerations

1. **Current ml-matrix usage**:
   - Already using for eigendecomposition in SpectralClustering
   - Proven 10x faster than our Jacobi implementation
   - Pure JavaScript, no native dependencies

2. **Scenarios to evaluate**:
   - User can't install tfjs-node due to build tools/permissions
   - WebGL not available (headless Node.js)
   - WASM disabled due to security policies
   - ml-matrix is faster for specific operations

3. **Decision matrix**:

   ```txt
   If ml-matrix is:
   - Consistently faster for some operations → Keep for those
   - Good fallback when TF.js struggles → Keep as fallback
   - Rarely better → Remove to reduce bundle size
   - Much smaller than TF.js → Consider ml-matrix-only version
   ```
