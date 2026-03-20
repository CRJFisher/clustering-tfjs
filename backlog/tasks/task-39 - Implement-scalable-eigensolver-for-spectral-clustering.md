---
id: TASK-39
title: Implement scalable eigensolver for spectral clustering
status: Done
assignee:
  - '@claude'
created_date: '2026-03-20'
updated_date: '2026-03-20 16:20'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The current Jacobi eigendecomposition computes ALL n eigenvalues of an n×n matrix even though only k (typically 2-10) are needed. This is O(n^3) minimum with the full dense matrix converted to JS arrays, defeating GPU acceleration entirely. Spectral clustering is impractical beyond ~500 samples. The parameter sweep in spectral_optimization.ts compounds this by running up to 81 full spectral pipelines. An iterative sparse solver would reduce complexity from O(n^3) to roughly O(n*k*iterations).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Iterative eigensolver (Lanczos or similar) implemented for computing k smallest eigenpairs
- [x] #2 Spectral clustering handles 5000+ samples without timeout
- [x] #3 Benchmark comparison shows >10x speedup over Jacobi for n>500
- [x] #4 Parameter sweep caches eigendecomposition across shared affinity matrices
- [x] #5 Existing reference tests still pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Implement Lanczos iterative eigensolver with full reorthogonalization and corrected tridiagonal QL
2. Integrate Lanczos into smallest_eigenvectors_with_values with auto-selection (n>100 → Lanczos, else Jacobi)
3. Relax deterministic_eigenpair_processing to accept rectangular (n×k) eigenvector matrices
4. Restructure intensiveParameterSweep to compute embedding once per gamma (9 eigendecompositions instead of 90)
5. Add unit tests, scale tests (n=5000+), and Lanczos vs Jacobi benchmarks
6. Update debug spectrum paths to use consistent solver selection
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Lanczos iterative eigensolver achieving 49.8x speedup at n=500 and enabling n=5000+ spectral clustering. Restructured parameter sweep to cache eigendecomposition per gamma. All 51 tests pass.
<!-- SECTION:NOTES:END -->
