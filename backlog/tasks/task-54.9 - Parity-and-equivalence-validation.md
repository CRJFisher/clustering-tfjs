---
id: TASK-54.9
title: Validate HDBSCAN parity and equivalence under the float32 front-half
status: To Do
assignee: []
created_date: '2026-06-24'
updated_date: '2026-06-24'
labels:
  - hdbscan
  - testing
dependencies:
  - task-54.6
  - task-54.7
  - task-54.8
parent_task_id: TASK-54
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

Confirm the migrated pipeline is correct end-to-end: labels match scikit-learn (and the 54.1 oracle) exactly up to permutation with consistent `-1` noise, and probabilities stay within the float32 bounds set in 54.2. Finalise the tie-free probability tolerance and the tie-bound MAE/agreement bounds from the _actual_ float32 outputs (54.2 set the structure and a provisional bound; this subtask tightens it to observed drift plus slack).

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [ ] #1 Labels from the migrated pipeline match the 54.1 oracle exactly (up to cluster-id permutation, consistent noise) on every fixture, including the degenerate all-noise and single-blob cases
- [ ] #2 Probabilities are within the 54.2 tolerances; the tie-free bound is finalised from the maximum observed float32 drift plus documented slack, and the tie-bound MAE/agreement bounds are confirmed against the float32 outputs (precomputed-cosine fixture checked explicitly)
- [ ] #3 The full suite (`hdbscan.test.ts`, `kdistance.test.ts`, `condensation_tree.test.ts`, `minimum_spanning_tree.test.ts`) passes
- [ ] #4 ESLint passes with no new errors (fix, do not ignore) before commit

<!-- AC:END -->

## Implementation Notes

Use `test_support/label_agreement.ts` (`alignment_agreement`, `labels_equivalent_with_noise`) as the oracle comparator. Treat a label mismatch versus the oracle as a real regression to fix (likely a gram-matrix cancellation flip — see 54.3's fallback), not a reason to weaken label parity. Only probability bounds may move, and only to track measured float32 drift.
