---
id: TASK-52.7
title: Fix SpectralClustering dispose() not clearing medoid_indices_
status: Done
assignee: []
created_date: '2026-06-10 08:56'
labels:
  - bug
  - plausible
dependencies: []
references:
  - 'src/clustering/spectral.ts:127'
parent_task_id: TASK-52
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

SpectralClustering.dispose() resets labels* and affinity matrices but does not clear medoid_indices*. A consumer that calls fit(data1), compute_medoids(), then fit(data2) will read stale medoid indices from the first fit until compute_medoids() is explicitly called again — the indices are into data1's label space but are silently presented as valid for data2.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [ ] #1 dispose() sets medoid*indices* to null alongside the other per-fit state
- [ ] #2 A re-fit after dispose() leaves medoid*indices* null until compute_medoids() is called again
- [x] #3 A test verifies that all per-fit output fields (labels*, affinity_matrix*, medoid*indices*) are null immediately after fit(data2) completes and before any accessor is called (complete per-fit state-reset test)
<!-- AC:END -->

## Implementation Notes

## High-level summary

`SpectralClustering.dispose()` manages all per-fit caches (`labels_`, `affinity_matrix_`, `sparse_affinity_matrix_`) but was missing a reset of `medoid_indices_`. Because `fit()` calls `dispose()` at the start, a consumer calling `fit(data1)` → `compute_medoids(data1)` → `fit(data2)` would silently carry forward medoid indices that index into data1's label space while appearing valid for data2.

The fix is one line: `this.medoid_indices_ = null` added to `dispose()`. Since `fit()` and `fit_with_intermediate_steps()` both open by calling `dispose()`, both paths benefit automatically.

A secondary issue surfaced during testing: the three `_debug_*` properties set inside `fit()` via `Object.defineProperty` used `configurable: false`, which caused `TypeError: Cannot redefine property` on any second `fit()` call on the same instance. These were changed to `configurable: true` (and `writable: true`) to allow re-fit — this change is a prerequisite for the re-fit scenario the acceptance criteria require.

The test suite adds three cases pinning the full lifecycle: an explicit `dispose()` test asserting all four per-fit fields go null, an AC#2 re-fit test checking `medoid_indices_` stays null, and an AC#3 state-completeness test asserting `labels_` and `affinity_matrix_` are populated while `sparse_affinity_matrix_` and `medoid_indices_` remain null after re-fit on a dense-affinity dataset.
