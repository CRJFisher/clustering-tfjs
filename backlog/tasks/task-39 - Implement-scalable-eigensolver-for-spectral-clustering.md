---
id: TASK-39
title: Implement scalable eigensolver for spectral clustering
status: In Progress
assignee:
  - '@claude'
created_date: '2026-03-20'
updated_date: '2026-03-20 16:19'
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
Implemented a Lanczos iterative eigensolver that reduces spectral clustering eigendecomposition from O(n³) to O(n²·m) where m is the Lanczos subspace size (typically 30-50 for k=3-10 clusters).

**Core algorithm** (`src/utils/lanczos.ts`):
- Standard Lanczos tridiagonalization with full reorthogonalization (double Gram-Schmidt when norm drops below 1/√2)
- Corrected tridiagonal QL solver (fixes variable-shadowing bug in eigen_qr.ts where `g = c*r - b` line was missing)
- Simple restart strategy using best Ritz vector as new starting point
- Convergence detection via residual bounds |β_m · s_i_last| < tol · max(1, |θ_i|)
- Float32-tuned tolerances (convergence 1e-6, breakdown 1e-10, PSD clamp 1e-5)
- Deterministic random starting vector via `make_random_stream`

**Integration** (`src/utils/smallest_eigenvectors_with_values.ts`):
- Auto-selects solver: Lanczos for n > 100 (and k < n/3), Jacobi fallback for small matrices
- Graceful fallback: if Lanczos fails (NaN, non-convergence), falls back to Jacobi with a warning
- Relaxed `deterministic_eigenpair_processing` in eigen_post.ts to accept rectangular (n×k) eigenvector matrices

**Parameter sweep caching** (`src/clustering/spectral_optimization.ts`):
- Restructured `intensiveParameterSweep()` to compute affinity+embedding once per gamma value
- Previously computed 90 eigendecompositions (9 gammas × 10 calls each), now only 9
- Uses try/finally for guaranteed tensor cleanup

**Performance results**:
- n=500: 49.8x speedup (Jacobi 5991ms → Lanczos 120ms)
- n=1000: Lanczos 117ms (Jacobi would take ~30-60s)
- n=5000: Spectral clustering completes in ~10s (previously infeasible)

**Test results**: 51 tests pass across 10 test suites, including all 12 spectral reference tests (ARI ≥ 0.95), 11 Lanczos unit tests, 2 scale tests (n=1000, n=5000), and 3 benchmark tests.

Modified files: src/utils/lanczos.ts (new), src/utils/smallest_eigenvectors_with_values.ts, src/utils/eigen_post.ts, src/utils/laplacian.ts, src/clustering/spectral.ts, src/clustering/spectral_optimization.ts, src/utils/platform.ts (pre-existing type fix)
Added files: test/utils/lanczos.test.ts, test/clustering/spectral_scale.test.ts, test/benchmarks/eigensolver_benchmark.test.ts
<!-- SECTION:NOTES:END -->
