---
id: TASK-52.5
title: Fix HDBSCAN dispose() called before empty-input validation guard
status: To Do
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

## Acceptance Criteria

<!-- AC:BEGIN -->

- [ ] #1 Input validation (including the n===0 guard) runs before dispose() is called
- [ ] #2 Calling fit([]) after a successful fit(data) preserves the model's previous labels*, probabilities*, and exemplar*indices*
- [ ] #3 A test verifies that a failed re-fit (empty input, then n < min*samples input) leaves labels*, probabilities*, and exemplar_indices* unchanged from the prior successful fit (state-preservation test covering multiple error conditions)
<!-- AC:END -->
