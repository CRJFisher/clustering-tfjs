---
id: TASK-52.3
title: >-
  Fix fit_with_intermediate_steps max_samples guard missing nearest_neighbors
  bypass
status: To Do
assignee: []
created_date: '2026-06-10 08:55'
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

- [ ] #1 fit_with_intermediate_steps applies the same max_samples bypass as fit() for affinity='nearest_neighbors'
- [ ] #2 fit_with_intermediate_steps uses the sparse affinity matrix for affinity='nearest_neighbors', matching fit()'s behavior
- [ ] #3 A test calls both fit() and fit_with_intermediate_steps() with affinity='nearest_neighbors' on a dataset with n > max_samples (default 10,000) and asserts both succeed and produce the same affinity matrix shape and type (entry-point equivalence test)
<!-- AC:END -->
