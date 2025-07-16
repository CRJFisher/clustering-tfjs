---
id: task-6
title: Implement linkage criteria for AgglomerativeClustering
status: Done
assignee: [@assistant]
created_date: '2025-07-15'
labels: []
dependencies: []
---

## Description

Implement the core linkage criteria algorithms (ward, complete, average, single) that determine how clusters are merged in the hierarchical clustering process

## Acceptance Criteria

- [x] Single linkage criterion implemented and tested
- [x] Complete linkage criterion implemented and tested
- [x] Average linkage criterion implemented and tested
- [x] Ward linkage criterion implemented with variance calculations
- [x] Distance matrix update logic for each linkage type
- [x] Unit tests validating linkage calculations
- [x] Integration tests with small example datasets

## Implementation Plan

1. Design API for linkage update independent from Agglomerative class.
2. Implement Lance–Williams recurrences for four linkage types.
3. Mutate distance matrix in place and maintain cluster size bookkeeping.
4. Write unit tests asserting distance updates, symmetry and bookkeeping.

## Implementation Notes

- Added `src/clustering/linkage.ts` with `update_distance_matrix` supporting
  single, complete, average and Ward linkage; Ward uses variance-based
  formula for numerical stability.
- Function works on simple `number[][]` avoiding TensorFlow overhead.
- Added `test/linkage.test.ts` covering all linkage criteria on toy dataset;
  also validates matrix contraction and cluster size updates.
- Documentation improved, checked edge cases (i>j handling, self-merge).
