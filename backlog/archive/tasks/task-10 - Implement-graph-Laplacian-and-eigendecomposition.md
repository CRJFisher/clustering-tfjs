---
id: task-10
title: Implement graph Laplacian and eigendecomposition
status: Done
assignee: []
created_date: '2025-07-15'
labels: []
dependencies: []
---

## Description

Compute the normalized graph Laplacian and perform eigendecomposition to create the spectral embedding, which is the core of spectral clustering

## Acceptance Criteria

- [x] Degree matrix calculation from affinity matrix
- [x] Normalized Laplacian computation (L_sym = I - D^(-1/2) A D^(-1/2))
- [x] Eigendecomposition (Jacobi solver fallback due to missing tf.linalg.eig)
- [x] Eigenvalue/eigenvector sorting implementation
- [x] Selection of k smallest eigenvalues and corresponding eigenvectors
- [x] Numerical stability checks and handling
- [x] Unit tests validating mathematical correctness

## Implementation Plan

1. Create new utility module `src/utils/laplacian.ts`.
2. Implement `degree_vector` and `normalised_laplacian` in pure TF ops.
3. Add Jacobi eigen-decomposition (since TFJS lacks `eig` on Node).
4. Provide helper `smallest_eigenvectors` to extract k smallest vectors.
5. Re-export helpers via `src/index.ts`.
6. Add comprehensive unit tests verifying each step.

## Implementation Notes

• Used `tf.where` to safely invert degree entries; isolates with zero degree stay identity in Laplacian.
• Jacobi solver operates on JS arrays for simplicity; sufficient for typical Spectral Clustering sizes and unit tests (< few hundred).
• Eigen-pairs sorted ascending; helper returns requested subset as `tf.Tensor2D`.
• Added numerical guard (tau===0) in rotation computation to avoid division by 0.
• Unit tests cover degree vector, Laplacian symmetry & expected values, eigenvalues (0 & 2 for 2-node graph) and magnitude-agnostic eigenvector check.
• All existing test suites pass (`npm test`).
