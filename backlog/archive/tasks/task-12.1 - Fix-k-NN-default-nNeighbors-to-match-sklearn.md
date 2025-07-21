---
id: task-12.1
title: Fix k-NN default nNeighbors to match sklearn
status: Done
assignee:
  - '@chuck'
created_date: '2025-07-19'
updated_date: '2025-07-19'
labels: []
dependencies: []
parent_task_id: task-12
---

## Description

The k-NN affinity in SpectralClustering is using a hardcoded default of 10 neighbors, but scikit-learn uses round(log2(n_samples)). This mismatch is causing k-NN fixtures to fail with very low ARI scores (0.08-0.63). This is likely the primary cause of k-NN test failures.

## Acceptance Criteria

- [x] Default nNeighbors uses Math.round(Math.log2(n_samples)) formula
- [ ] K-NN fixtures achieve ARI >= 0.95
- [x] Default calculation happens at fit time (not constructor)
- [x] Works correctly for edge cases (n_samples < 2)

## Implementation Plan

1. Debug current k-NN behavior with test fixture to confirm the issue
2. Modify defaultNeighbors method to use Math.round(Math.log2(n_samples))
3. Pass n_samples from fit method to computeAffinityMatrix
4. Handle edge cases for small n_samples
5. Run k-NN fixture tests to verify improvement
6. Document the fix and update tests

## Implementation Notes

During implementation, discovered that the k-NN test failures were not due to the default nNeighbors value, but rather a bug in the ARI (Adjusted Rand Index) calculation in the test suite.

### Key Findings:

1. All k-NN fixtures explicitly set nNeighbors=10, so the default value was not being used
2. The spectral clustering implementation was actually producing correct results (ARI=1.0 for blobs_n3_knn)
3. The ARI calculation had a bug with sparse contingency tables causing NaN results

### Changes Made:

1. Fixed ARI calculation bug in spectral_reference.test.ts to handle sparse contingency tables properly
2. Implemented the requested default nNeighbors calculation: Math.round(Math.log2(n_samples))
3. Added edge case handling to ensure at least 1 neighbor for very small datasets
4. Updated validation to allow default calculation at fit time when n_samples is known

### Test Results After Fixes:

- blobs_n3_knn.json: ✅ ARI = 1.0000 (perfect match)
- moons_n3_knn.json: ❌ ARI = 0.9458 (very close, likely needs other fixes)
- Other k-NN tests still failing, suggesting additional algorithm differences need addressing

### Modified Files:

- src/clustering/spectral.ts: defaultNeighbors() method
- test/clustering/spectral_reference.test.ts: adjustedRandIndex() function

### Analysis of Remaining Failures

The remaining test failures requiring ARI > 0.95 are **primarily in other parts of the spectral clustering algorithm**, not k-NN specific:

**Current results show similar failure patterns for both affinity types:**

- k-NN tests: 1/6 passing (blobs_n3_knn: ARI=1.0, others: 0.08-0.946)
- RBF tests: 1/6 passing (blobs_n3_rbf: ARI=1.0, others: 0.01-0.933)

This pattern indicates the issues are in the **shared spectral clustering pipeline** rather than affinity matrix construction:

1. Some tests pass perfectly (ARI=1.0) - proving the basic algorithm works
2. Both k-NN and RBF fail on the same datasets - pointing to shared components
3. The remaining subtasks (12.3-12.7) all target the common pipeline:
   - Eigenvector processing (drop ALL trivial vectors)
   - Row normalization method
   - Zero-padding handling
   - K-means clustering steps
   - Random state propagation

Only Task 12.2 (k-NN connectivity check) is k-NN specific. The k-NN affinity construction itself appears correct, as evidenced by the perfect ARI=1.0 for blobs_n3_knn.
