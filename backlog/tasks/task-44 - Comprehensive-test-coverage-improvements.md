---
id: TASK-44
title: Comprehensive test coverage improvements
status: In Progress
assignee:
  - '@claude'
created_date: '2026-03-20'
updated_date: '2026-03-20 21:28'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The test suite has significant coverage gaps and quality issues identified across all reviewers. Major untested modules: connected_components.ts, component_indicators.ts, spectral_consensus.ts, spectral_optimization.ts, eigen_improved.ts, eigen_qr.ts, som_visualization.ts (6 exports), platform.ts, tf-backend.ts, datasets/synthetic.ts. Test quality issues: Math.random() without seed in 6+ test files causing flaky tests, incorrect afterEach scope cleanup in all 3 validation test files (startScope/endScope is a no-op), areLabelingsEquivalent missing reverse-mapping check (undermines 13 reference tests), only 2 of 12 SOM fixtures tested, very loose tolerance thresholds (50% for QE, 0.3 for U-matrix correlation).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Unit tests added for connected_components and component_indicators
- [x] #2 Tests added for spectral_consensus and spectral_optimization
- [x] #3 Tests added for eigen_improved and eigen_qr including degenerate eigenvalue cases
- [x] #4 Tests added for som_visualization functions
- [x] #5 All test data generation uses seeded RNG not Math.random()
- [x] #6 Validation test afterEach uses correct beforeEach-startScope afterEach-endScope pattern
- [x] #7 areLabelingsEquivalent checks both forward and reverse mapping
- [x] #8 SOM reference tests exercise all 16 fixtures not just first 2
- [x] #9 KMeans tests added for K=1 and K=nSamples and duplicate points and tensor input
- [x] #10 Integration test added that verifies findOptimalClusters return shape matches declared TypeScript types (from code-charter integration issue #1)
<!-- AC:END -->

## Implementation Plan

1. Fix test quality issues (AC#5, #6, #7, #8) in existing files
2. Create new unit test files for untested modules (AC#1, #2, #3, #4)
3. Add edge case tests to existing test files (AC#9)
4. Create integration test for return shape validation (AC#10)
5. Run full suite and linter, fix any issues
6. Review with 5 opus reviewer agents and address findings

## Implementation Notes

### Approach
Used 5 parallel opus planner agents to research each area, then 5 parallel opus agents to write all test files simultaneously. After initial commit, ran 5 parallel opus reviewer agents to verify the work and fixed all findings.

### Files Created (7 new test files)
- `test/utils/connected_components.test.ts` - 12 tests (BFS component detection, connectivity check)
- `test/utils/component_indicators.test.ts` - 10 tests (indicator matrix, normalization, capping)
- `test/utils/eigen_improved.test.ts` - 12 tests (Jacobi solver, degenerate eigenvalues, laplacian decomposition)
- `test/utils/eigen_qr.test.ts` - 11 tests (QR solver, tridiagonal QL, degenerate cases; 3 marked `.failing` for known QR NaN bug)
- `test/utils/som_visualization.test.ts` - 19 tests (all 8 exported functions)
- `test/clustering/spectral_optimization.test.ts` - 8 tests (validationBasedOptimization)
- `test/integration/findOptimalClusters_shape.test.ts` - 6 tests (runtime shape validation)

### Files Modified (12 existing test files)
- 6 test files: replaced Math.random() with make_random_stream(42)
- 5 test files: split broken startScope+endScope from single afterEach into proper beforeEach/afterEach
- `test/clustering/agglomerative_reference.test.ts`: added reverse mapping to areLabelingsEquivalent
- `test/clustering/som.reference.test.ts`: removed .slice(0, 2) from all 4 describe blocks, added jest.setTimeout(120s)
- `test/clustering/spectral_consensus.test.ts`: added 5 new behavior tests
- `test/kmeans.test.ts`: added 10 edge case tests (K=1, K=nSamples, duplicates, tensor input)

### Key Findings
- `qr_eigen_decomposition` has a real bug: Wilkinson shift produces NaN for non-diagonal matrices. Tests documented with `it.failing`.
- SOM fixture directory has 16 fixtures (not 12 as originally counted in AC#8)
- Pre-existing test failures in lanczos.test.ts and spectral_reference.test.ts unrelated to our changes
