---
id: task-5
title: Implement pairwise distance computation for AgglomerativeClustering
status: Done
assignee: []
created_date: '2025-07-15'
labels: []
dependencies: []
---

## Description

Implement the pairwise distance matrix calculation for AgglomerativeClustering supporting multiple distance metrics using TensorFlow.js operations

## Acceptance Criteria

- [x] Pairwise distance function implemented for euclidean metric
- [x] Support added for manhattan distance metric
- [x] Support added for cosine distance metric
- [x] Efficient tensor broadcasting used for computation
- [x] Memory-efficient implementation using tf.tidy()
- [x] Unit tests comparing output with reference implementations
- [x] Performance tests for different matrix sizes

## Implementation Plan

1. Extend tensor utilities with a general `pairwiseDistanceMatrix` API.
2. Re-use existing optimised Euclidean implementation; add Manhattan & Cosine via broadcasting / Gram tricks.
3. Wrap calculations in `tf.tidy` and enforce symmetry & zero diagonal for numerical robustness.
4. Write unit tests for correctness vs. naive loops and smoke performance test.

## Implementation Notes

- Added `pairwiseDistanceMatrix` to `src/utils/tensor.ts` supporting `euclidean`, `manhattan`, and `cosine` metrics.
- Ensured memory safety with `tf.tidy` and numerical stability by enforcing symmetry and zeroed diagonal.
- Created `test/utils/pairwise.test.ts` covering correctness for all metrics and a 100×50 performance smoke check.
- All Jest tests pass, confirming criteria met.
