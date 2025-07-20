# Task 12 Restructuring After 12.10 Investigation

## Key Discovery from Task 12.10

The root issue is that our eigenvectors have different numerical values than sklearn's, even though they have the correct structure (same zero/non-zero pattern for disconnected components). This suggests the problem is in:

1. How the normalized Laplacian is computed
2. How eigenvectors are computed/scaled
3. Possible numerical differences in the eigensolver

## Tasks to Remove

### task-12.15 - Debug k-means initialization differences

- **Reason**: K-means++ is already correctly implemented
- **Action**: Delete - not the source of issues

### task-12.16 - Investigate affinity matrix sparsity handling

- **Reason**: Dense implementation works correctly; sparse is just optimization
- **Action**: Delete - not relevant to correctness

### task-12.18 - Investigate 2-cluster vs 3-cluster performance difference

- **Reason**: This is explained by the eigenvector scaling issue discovered in 12.10
- **Action**: Delete - already understood

## Tasks to Deprioritize

### task-12.11 - Fix RBF gamma scaling for fixture data

- **Reason**: Secondary issue; fix eigenvectors first
- **Action**: Keep but move to lower priority

### task-12.12 - Complete random state propagation

- **Reason**: Good practice but not causing test failures
- **Action**: Keep but move to lower priority

### task-12.13 - Handle disconnected graph components

- **Reason**: Edge case; main issue is eigenvector values
- **Action**: Keep but move to lower priority

### task-12.17 - Improve spectral clustering for overlapping clusters

- **Reason**: Symptom not cause; fix eigenvectors first
- **Action**: Keep but move to lower priority

## New High Priority Tasks to Create

### NEW task-12.19 - Fix eigenvector computation to match sklearn

**Description**: Our eigenvectors have the correct structure but different values than sklearn's. Investigate and fix the computation differences.
**Acceptance Criteria**:

- Compare our Laplacian computation with sklearn/scipy line by line
- Check if eigenvector post-processing/scaling differs
- Implement fixes to match sklearn's eigenvector values
- Achieve same 3-unique-values pattern for disconnected component cases

### NEW task-12.20 - Investigate eigensolver differences with sklearn

**Description**: sklearn uses ARPACK with shift-invert mode while we use Jacobi. This might cause numerical differences.
**Acceptance Criteria**:

- Document sklearn's exact eigensolver approach
- Compare numerical results between solvers
- Determine if solver difference is causing value discrepancies
- Implement compatible solution if needed

## Task to Refocus

### task-12.14 - Test with sklearn's exact parameters and data

- **Refocus**: Specifically on comparing eigenvector values at each step
- **Update AC**: Add detailed eigenvector value comparison

## Recommended Execution Order

1. **task-12.19** - Fix eigenvector computation (CRITICAL - root cause)
2. **task-12.20** - Investigate eigensolver differences (if 12.19 doesn't fully resolve)
3. **task-12.14** - Comprehensive sklearn comparison (validation)
4. **task-12.11** - Fix RBF gamma scaling (improve RBF results)
5. **task-12.13** - Handle disconnected components (robustness)
6. **task-12.12** - Complete random state propagation (determinism)
7. **task-12.17** - Improve overlapping clusters (if still needed)

## Expected Outcome

Fixing the eigenvector computation (12.19) should resolve the blobs_n2 failure and likely improve other test results significantly. This is the most critical issue identified.
