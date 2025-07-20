# Restructured SpectralClustering Tasks Based on Investigation Results

## Current Status: 6/12 tests passing (50%)

After completing tasks 12.1-12.8 and 12.11, we've made significant progress but key issues remain. Based on our findings, here's a restructured plan for the remaining work.

## Critical Discovery from Task 12.11

The k-NN symmetrization bug fix (changing from `max(A, A^T)` to `0.5 * (A + A^T)`) improved results from 5/12 to 6/12 passing tests. This shows that subtle implementation differences have major impacts.

## Restructured Priority Tasks (In Implementation Order)

### Priority 1: Debug Two-Cluster Special Case

**Task 12.10 - Debug two-cluster special case**
- **Rationale**: Both blobs_n2 tests have ARI = 0.088, suggesting a fundamental n=2 issue
- **Approach**:
  1. Create minimal reproduction with 2-cluster blobs data
  2. Compare eigenvector selection for n=2 vs n=3
  3. Check if sklearn has special handling for binary clustering
  4. Verify k-means behaves correctly with 2D embeddings
- **Success Metric**: blobs_n2_* tests achieve ARI > 0.95

### Priority 2: RBF Affinity Deep Dive

**Task 12.11 - Fix RBF gamma scaling for fixture data**
- **Rationale**: Task 12.5 discovered fixtures use gamma=1.0 but need ~0.1-0.7
- **Approach**:
  1. Update test fixtures with correct gamma values
  2. OR implement gamma auto-scaling based on data variance
  3. Test RBF performance with properly scaled gamma
- **Success Metric**: RBF tests perform comparably to k-NN tests

### Priority 3: Complete Pipeline Determinism

**Task 12.12 - Complete random state propagation**
- **Rationale**: Some randomness may remain causing test instability
- **Approach**:
  1. Audit all random operations in pipeline
  2. Add determinism tests with multiple seeds
  3. Ensure tie-breaking is consistent
- **Success Metric**: Identical results for same random seed

### Priority 4: Disconnected Components Handling

**Task 12.13 - Handle disconnected graph components**
- **Rationale**: Debug output shows some graphs have multiple components
- **Approach**:
  1. Add connected components check before spectral embedding
  2. Implement sklearn's approach for disconnected graphs
  3. Test with artificially disconnected data
- **Success Metric**: Robust handling of disconnected cases

### Priority 5: Final sklearn Comparison

**Task 12.14 - Final comprehensive sklearn comparison**
- **Rationale**: Need systematic comparison after all fixes
- **Enhanced Approach**:
  1. Create side-by-side comparison framework
  2. Test with both fixture data AND synthetic data
  3. Profile numerical differences at each step
  4. Document any remaining acceptable differences
- **Success Metric**: Understand and document all differences

## Deprioritized/Removed Tasks

### Task 12.8 - Float64 precision
- **Status**: COMPLETED - Won't Do (TensorFlow.js limitation)
- **Action**: Already marked as Won't Do

### Task 12.15 - Debug k-means initialization (was 12.9)
- **Status**: DEPRIORITIZED
- **Rationale**: K-means++ already implemented and working correctly
- **Action**: Moved to lower priority

### Task 12.16 - Investigate affinity matrix sparsity (was 12.13)
- **Status**: DEPRIORITIZED  
- **Rationale**: Current dense implementation works; sparsity is optimization
- **Action**: Moved to future optimization work

## Recommended Execution Order

1. **Task 12.10**: Debug two-cluster special case - Most likely to unlock multiple test fixes
2. **Task 12.11**: Fix RBF gamma scaling - Simple fix with high impact
3. **Task 12.12**: Complete random state propagation - Ensure consistent results
4. **Task 12.13**: Handle disconnected graph components - Handle edge cases
5. **Task 12.14**: Final comprehensive sklearn comparison - Verify all improvements

## Expected Outcomes

With the k-NN symmetrization fix already improving results to 50% passing, addressing the two-cluster issue and RBF scaling should get us to 75%+ passing tests. The remaining issues are likely edge cases that can be documented as acceptable differences from sklearn.

## Key Lessons Applied

1. **Implementation details matter**: The k-NN symmetrization fix shows tiny details have huge impacts
2. **Test fixtures may have issues**: The gamma=1.0 problem shows we must validate test data
3. **Focus on patterns**: Two-cluster failures across affinity types suggest algorithmic issues
4. **Numerical precision isn't everything**: Float64 investigation showed it's not the bottleneck

## Next Immediate Action

Start with Task 12.10 focusing specifically on why two-cluster cases fail catastrophically (ARI = 0.088). This is likely a simple fix that will unlock multiple test improvements.