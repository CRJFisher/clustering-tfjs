---
id: TASK-52.3
title: >-
  Fix fit_with_intermediate_steps max_samples guard missing nearest_neighbors
  bypass
status: Done
assignee:
  - crjfisher
created_date: '2026-06-10 08:55'
updated_date: '2026-06-10 13:40'
labels:
  - bug
  - confirmed
dependencies: []
references:
  - 'src/clustering/spectral.ts:573'
parent_task_id: TASK-52
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
SpectralClustering.fit() skips the max_samples guard when affinity='nearest_neighbors' (using a sparse path instead). fit_with_intermediate_steps applies the guard unconditionally and also materialises a dense O(n²) affinity matrix for nearest_neighbors rather than the sparse matrix that fit() produces. These two divergences mean the debug path rejects inputs the main path accepts, and runs a fundamentally different computation when it doesn't reject.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 fit_with_intermediate_steps applies the same max_samples bypass as fit() for affinity='nearest_neighbors'
- [x] #2 fit_with_intermediate_steps uses the sparse affinity matrix for affinity='nearest_neighbors', matching fit()'s behavior
- [x] #3 A test calls both fit() and fit_with_intermediate_steps() with affinity='nearest_neighbors' on a dataset with n > max_samples (default 10,000) and asserts both succeed and produce the same affinity matrix shape and type (entry-point equivalence test)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add `use_sparse_nearest_neighbors` flag before the max_samples guard in `fit_with_intermediate_steps`.
2. Gate the max_samples guard with `!use_sparse_nearest_neighbors`, matching `fit()`.
3. Branch the affinity computation: for nearest_neighbors use `compute_sparse_knn_affinity`, store in `this.sparse_affinity_matrix_`, then materialize to dense via `sparse_to_dense_tensor` for the rest of the pipeline.
4. Add import of `sparse_to_dense_tensor` from `../graph/sparse`.
5. Update affinity_sum check and affinity_stats capture to use the sparse path when available.
6. Conditionally skip setting `this.affinity_matrix_` for the sparse path (matching `fit()`).
7. Add the entry-point equivalence test to `spectral_sparse.test.ts`.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added `use_sparse_nearest_neighbors` flag in `fit_with_intermediate_steps` to mirror the bypass in `fit()`. The max_samples guard is now gated with `!use_sparse_nearest_neighbors`. For the sparse path, `compute_sparse_knn_affinity` is called directly (matching `fit()`), the result is stored in `this.sparse_affinity_matrix_`, and then materialized to a dense `tf.Tensor2D` via `sparse_to_dense_tensor` for the remainder of the debug pipeline (Laplacian, embedding, return value). `this.affinity_matrix_` is left null for the sparse path, matching `fit()`'s state semantics. The affinity_sum check and affinity_stats capture also branch on the sparse path using the existing `sparse_stats` helper. Added `sparse_to_dense_tensor` to the import from `../graph/sparse`. Added an entry-point equivalence test in `spectral_sparse.test.ts` that calls both `fit()` and `fit_with_intermediate_steps()` with `n > max_samples` and asserts both succeed with identical sparse shape and null dense affinity_matrix_.

Post-review fixes: (1) Added affinity.dispose() + x_tensor.dispose() before the zero-affinity throw to prevent tensor leak on the nearest_neighbors path (memory reviewer finding). (2) Strengthened the equivalence test to compare sparse_affinity_matrix_.indices and .data between both models, not just shape/null-state (test quality finding).
<!-- SECTION:NOTES:END -->
