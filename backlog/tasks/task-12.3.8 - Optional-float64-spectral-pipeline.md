---
id: task-12.3.8
title: Optional float64 spectral embedding pipeline
status: To Do
assignee: []
created_date: '2025-07-17'
labels: []
dependencies: [task-12.3.7]
---

## Description (the why)

Several k-NN fixtures show ARI < 0.95 due to tiny eigenvalue gaps that cause
column swaps in float32 precision.  scikit-learn performs its eigen-decomposition
in float64 by default.  Allowing our pipeline to run in float64 eliminates the
numerical instability and aligns results.

## Acceptance Criteria (the what)

- [ ] `SpectralClusteringParams` gains optional `dtype?: "float32" | "float64"` (default "float32").
- [ ] `computeAffinityMatrix`, Laplacian construction and Jacobi solver honour the requested dtype.
- [ ] Unit test: for a crafted matrix with two near-equal eigenvalues, float32 swaps columns in ≥50 % of runs while float64 remains stable.
- [ ] All spectral fixtures pass (ARI ≥ 0.95) with `dtype="float64"` and their original parameters.
- [ ] Documentation updated to explain performance/precision trade-off.

## Implementation Plan (the how)

1. Thread a `dtype` parameter through `SpectralClustering` constructor → affinity helpers → Laplacian → Jacobi.
2. Where TensorFlow.js kernels lack float64 support on the backend, fall back to float32 and issue a warning.
3. Add CLI flag `--float64` to `scripts/debug_spectral_parity.ts` for quick experimentation.

## Dependencies

Optional; can be implemented in parallel with task-12.3.7.

## Implementation Notes

*To be filled after completion*