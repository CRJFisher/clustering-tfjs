# Task 12 - Refocused Plan for sklearn Parity

## Problem Statement

SpectralClustering is failing to achieve parity with scikit-learn, particularly for k-NN affinity (ARI 0.08-0.63).

## Root Causes Identified

### 1. k-NN Affinity Issues (CRITICAL)

- **Wrong default**: Using `nNeighbors=10` instead of sklearn's `round(log2(n_samples))`
- **Missing connectivity enforcement**: No self-connections for disconnected graphs
- **Possible affinity matrix construction differences**

### 2. Algorithm Differences

- Only dropping first trivial eigenvector instead of ALL
- Different row normalization: `norm + eps` vs `max(norm, eps)`
- Zero-padding embedding when insufficient eigenvectors
- Different k-means empty cluster handling
- Incomplete randomState propagation

### 3. Minor Issues

- Some numerical precision differences (float32 vs float64)
- Edge cases in deterministic tie-breaking

## Prioritized Fix Order

### IMMEDIATE (Fix k-NN first!)

1. Debug k-NN affinity construction:

   ```bash
   # Run debug script on k-NN fixture
   node debug_fixture.ts test/fixtures/spectral/blobs_n3_knn.json
   ```

2. Fix default nNeighbors:

   ```typescript
   // In spectralClustering constructor or computeAffinityMatrix
   const defaultNeighbors = Math.round(Math.log2(n_samples));
   ```

3. Check k-NN graph connectivity and add self-loops if needed

### NEXT (Core algorithm fixes)

1. Implement task-12.1 through task-12.5 from backlog/tasks IN ORDER
2. Each fix is small and targeted - complete one at a time
3. Test after each fix to measure progress

### LATER (Only if needed)

- Float64 mode
- Additional determinism refinements

## Success Criteria

- ALL fixtures pass with ARI ≥ 0.95
- Both RBF and k-NN affinity modes work correctly
- Results are deterministic with fixed randomState

## Next Steps

1. Start with k-NN debugging to understand the failure
2. Fix the default parameter issue
3. Work through the 5 pending algorithm fixes systematically
