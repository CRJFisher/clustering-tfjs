---
id: TASK-54.8
title: Memory discipline and dispose audit for the HDBSCAN tensor pipeline
status: To Do
assignee: []
created_date: '2026-06-24'
updated_date: '2026-06-24'
labels:
  - hdbscan
  - tfjs
  - memory
dependencies:
  - task-54.5
  - task-54.6
parent_task_id: TASK-54
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

HDBSCAN previously held no tensors. The front-half migration introduces a multi-stage tensor pipeline (distance, core distances, mutual-reachability) plus an input upload, so it must follow the library's tidy/dispose discipline established in task-35 (tensor memory leaks across estimators). Audit the new pipeline for leaked intermediates and confirm the estimator's lifecycle contracts still hold.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [ ] #1 All front-half intermediates are disposed (single `tf.tidy` for the fused stages, or explicit dispose of the input upload and any tensor that escapes a tidy); a `tf.memory()`-based test asserts no net tensor growth across repeated `fit()` calls
- [ ] #2 `dispose()` continues to reset `labels_`, `probabilities_`, and `exemplar_indices_` and holds no tensor instance state (HDBSCAN keeps no tensors between `fit` calls)
- [ ] #3 The empty-input (`n === 0`) guard, the `n === 1` graceful-noise path, and the `precomputed` path all behave exactly as before, with validation still running before `dispose()` so a failed re-fit preserves prior state
- [ ] #4 A large-`n` fit (consistent with the AGENTS.md "test at scale" practice, within the memory ceiling) runs without leaking and without argument-spread RangeErrors

<!-- AC:END -->

## Implementation Notes

Reuse the task-35 memory-regression pattern (`src/memory_regression.test.ts`). The single-readback design from 54.5 keeps the tensor lifetime short and bounded; the main audit targets are the input upload and the `tf.topk` outputs (indices tensor is unused and must be disposed). Keep the existing state-preservation invariant from task-52.5 intact.
