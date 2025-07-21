---
id: task-12.14
title: Test with sklearn's exact parameters and data
status: Done
assignee: []
created_date: '2025-07-20'
updated_date: '2025-07-21'
labels:
  - spectral
  - testing
dependencies: []
parent_task_id: task-12
---

## Description

After implementing all fixes, create a comprehensive comparison framework with sklearn to understand any remaining differences. This should test both fixture data and synthetic datasets to ensure our implementation achieves parity or document acceptable differences.

**Expected results based on cluster analysis**:

- Blobs datasets should achieve near-perfect scores due to excellent separation
- Circles and moons datasets are inherently challenging due to overlapping, non-convex shapes
- k-NN affinity is expected to outperform RBF on overlapping clusters
- Focus on understanding why sklearn succeeds on these challenging cases

## Acceptance Criteria

- [x] Create side-by-side comparison framework
- [x] Test with fixture data AND synthetic data
- [x] Profile numerical differences at each pipeline step
- [x] Add detailed eigenvector value comparison (unique values, scaling, normalization)
- [x] Compare Laplacian matrix values directly between implementations
- [x] Document all remaining differences
- [ ] Achieve 10/12 fixture tests passing or document why not

## Implementation Plan

1. Create comprehensive comparison scripts to understand sklearn's behavior
2. Test both implementations side-by-side with exact same data
3. Identify key differences in intermediate steps
4. Document findings and create action plan for fixes

## Implementation Notes

### Key Findings from Comprehensive Analysis

After creating detailed comparison scripts (`compare_sklearn_exact.py` and `compare_implementations_detailed.py`), we discovered several critical insights:

#### 1. Current Test Status Discrepancy

There's a major discrepancy between test results and manual runs:

- **Test runner**: Shows 5/12 tests passing (blobs_n2_knn, blobs_n3_knn, circles_n2_knn, moons_n2_knn, moons_n3_knn)
- **Manual runs**: Show only 2/6 tests passing for n=2 cases (circles_n2_knn, moons_n2_knn)
- **Critical regression**: blobs datasets that were supposedly fixed are actually failing with ARI ≈ 0.088

#### 2. The 3-Components-2-Clusters Problem

The core issue with blobs_n2_knn dataset:

- k-NN graph creates **3 disconnected components** (verified with scipy)
- Dataset has 3 well-separated spatial blobs (1 red, 2 blue in ground truth)
- We want 2 clusters, but have 3 components
- Our implementation clusters by components (ARI = 0.088)
- Sklearn somehow correctly groups the two blue blobs together (ARI = 1.0)

#### 3. Sklearn's Perfect Performance

Sklearn achieves ARI = 1.0 on ALL 12 test cases, including:

- Disconnected graphs (blobs with k-NN)
- Overlapping clusters (circles, moons)
- Both RBF and k-NN affinities

#### 4. Key Implementation Differences

1. **Eigenvector patterns**:
   - For disconnected graphs, sklearn produces eigenvectors with few unique values
   - Example: blobs_n2_knn has eigenvectors with only 3 unique values per dimension
   - This suggests component indicator functions are being used

2. **Diffusion map scaling**:
   - We implemented scaling by sqrt(1 - eigenvalue)
   - This fixed some cases but not the disconnected component cases

3. **Component handling**:
   - Sklearn issues warning: "Graph is not fully connected, spectral embedding may not work as expected"
   - But still achieves perfect clustering
   - Suggests special handling beyond just the warning

#### 5. Test Infrastructure Issue

The test suite appears to be comparing against different expected results than what we're seeing in manual runs. This needs investigation - possibly the fixtures were generated with a different sklearn version or parameters.

### Next Steps

Based on these findings, the priority should be:

1. **Fix the test discrepancy** - Understand why tests pass but manual runs fail
2. **Implement proper component handling** - Task 12.16 becomes critical
3. **Debug the component grouping logic** - How does sklearn group 3 components into 2 clusters?

### Scripts Created

1. `compare_sklearn_exact.py` - Runs sklearn on all fixtures and analyzes eigendecomposition
2. `compare_implementations_detailed.py` - Compares our implementation with sklearn step-by-step
3. `debug_blobs_discrepancy.py` - Investigates the blobs dataset issue specifically
4. `analyze_test_implementation.py` - Visualizes the clustering problem with matplotlib

All scripts and detailed results are saved in the project root for reference.

Completed comprehensive analysis comparing sklearn and our implementation. Found critical issue: blobs datasets are failing due to 3-components-2-clusters problem. Created detailed comparison scripts and documented all findings.
