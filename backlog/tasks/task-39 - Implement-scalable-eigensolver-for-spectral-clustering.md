---
id: task-39
title: Implement scalable eigensolver for spectral clustering
status: To Do
assignee: []
created_date: '2026-03-20'
labels: []
dependencies: []
---

## Description

The current Jacobi eigendecomposition computes ALL n eigenvalues of an n×n matrix even though only k (typically 2-10) are needed. This is O(n^3) minimum with the full dense matrix converted to JS arrays, defeating GPU acceleration entirely. Spectral clustering is impractical beyond ~500 samples. The parameter sweep in spectral_optimization.ts compounds this by running up to 81 full spectral pipelines. An iterative sparse solver would reduce complexity from O(n^3) to roughly O(n*k*iterations).

## Acceptance Criteria

- [ ] Iterative eigensolver (Lanczos or similar) implemented for computing k smallest eigenpairs
- [ ] Spectral clustering handles 5000+ samples without timeout
- [ ] Benchmark comparison shows >10x speedup over Jacobi for n>500
- [ ] Parameter sweep caches eigendecomposition across shared affinity matrices
- [ ] Existing reference tests still pass
