# Task 12 Current Priorities After Restructuring

## Key Discovery
The root issue with SpectralClustering is that our eigenvectors have different numerical values than sklearn's, even though they have the correct structure (same zero/non-zero pattern). This was discovered in task 12.10 when debugging the blobs_n2 failure.

## Execution Priority Order

### Critical Path (Must Fix)
1. **task-12.11** - Fix eigenvector computation to match sklearn
   - Root cause of blobs_n2 failure (ARI=0.088)
   - Our eigenvectors have many unique values vs sklearn's 3 unique values per column
   - Most likely to fix multiple test failures

2. **task-12.12** - Investigate eigensolver differences with sklearn
   - Only if 12.11 doesn't fully resolve the issue
   - sklearn uses ARPACK, we use Jacobi

3. **task-12.13** - Test with sklearn's exact parameters and data
   - Validation step after fixes
   - Now includes detailed eigenvector comparison

### Secondary Issues (Nice to Have)
4. **task-12.14** - Fix RBF gamma scaling for fixture data
   - Will improve RBF results but not critical

5. **task-12.15** - Handle disconnected graph components
   - Edge case handling for robustness

6. **task-12.16** - Complete random state propagation
   - For full determinism

7. **task-12.17** - Improve overlapping clusters
   - Only if still needed after eigenvector fixes

## Tasks Completed
- task-12.1 through task-12.10
- Major fix: k-NN symmetrization changed from max(A, A^T) to 0.5 * (A + A^T)
- Improved results from 5/12 to 6/12 passing tests

## Current Status
- 6/12 fixture tests passing
- Main blocker: eigenvector numerical differences (task-12.19)
- Once fixed, expect significant improvement in test results