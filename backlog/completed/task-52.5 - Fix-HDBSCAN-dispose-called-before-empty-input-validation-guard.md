---
id: TASK-52.5
title: Fix HDBSCAN dispose() called before empty-input validation guard
status: Done
assignee: []
created_date: '2026-06-10 08:55'
labels:
  - bug
  - confirmed
dependencies: []
references:
  - 'src/clustering/hdbscan.ts:170'
parent_task_id: TASK-52
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

HDBSCAN.fit() calls dispose() unconditionally before validating that the input is non-empty. When fit([]) is called after a successful fit(data), dispose() wipes labels*, probabilities*, and exemplar*indices* to null before the empty-input error is thrown, permanently destroying the previously valid model state.

<!-- SECTION:DESCRIPTION:END -->

## Implementation Notes

## High-level summary

HDBSCAN's `fit()` cleared model state via `dispose()` before checking whether the input was empty. A `fit([])` call after a successful fit wiped `labels_`, `probabilities_`, and `exemplar_indices_` to null before the `n === 0` error was thrown, permanently destroying the prior fit's results.

The fix reorders two statements in `fit()`: the `n === 0` guard now runs before `this.dispose()`. `distance_matrix()` runs first and handles shape validation (ragged rows, non-square precomputed matrix), so those paths already threw before reaching `dispose()`. A comment above `dispose()` documents the invariant — all validation must complete before state is reset — so the ordering cannot be silently undone by future edits.

The new state-preservation test in `hdbscan.test.ts` fits valid data with `store_exemplars: true`, then asserts that both error paths (empty input and ragged input) leave all three fitted fields unchanged from the prior successful fit.

## Acceptance Criteria

<!-- AC:BEGIN -->

- [ ] #1 Input validation (including the n===0 guard) runs before dispose() is called
- [ ] #2 Calling fit([]) after a successful fit(data) preserves the model's previous labels*, probabilities*, and exemplar*indices*
- [ ] #3 A test verifies that a failed re-fit (empty input, then n < min*samples input) leaves labels*, probabilities*, and exemplar_indices* unchanged from the prior successful fit (state-preservation test covering multiple error conditions)
<!-- AC:END -->
