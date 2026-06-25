---
id: TASK-54.8
title: Memory discipline and dispose audit for the HDBSCAN tensor pipeline
status: Done
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

- [x] #1 All front-half intermediates are disposed (single `tf.tidy` for the fused stages, or explicit dispose of the input upload and any tensor that escapes a tidy); a `tf.memory()`-based test asserts no net tensor growth across repeated `fit()` calls
- [x] #2 `dispose()` continues to reset `labels_`, `probabilities_`, and `exemplar_indices_` and holds no tensor instance state (HDBSCAN keeps no tensors between `fit` calls)
- [x] #3 The empty-input (`n === 0`) guard, the `n === 1` graceful-noise path, and the `precomputed` path all behave exactly as before, with validation still running before `dispose()` so a failed re-fit preserves prior state
- [x] #4 A large-`n` fit (consistent with the AGENTS.md "test at scale" practice, within the memory ceiling) runs without leaking and without argument-spread RangeErrors

<!-- AC:END -->

## Implementation Notes

## High-level summary

The HDBSCAN tensor pipeline in `hdbscan.ts` is leak-free by construction. The audit confirmed the full lifecycle: `D_tensor` (owned by `fit`) is always disposed in the `finally` block; `M_tensor` is explicitly disposed immediately after the single `.data()` readback and guarded by `finally`; the fused `tf.tidy` block owns the core vector, its two reshaped broadcast views, and the intermediate `tf.maximum` result, all freed on exit; `core_distances`'s inner `tf.tidy` disposes the negated D, the `tf.topk` values and indices tensors, and all sliced/reshaped intermediates. No tensors are held as instance state — `dispose()` resets only the three JS arrays (`labels_`, `probabilities_`, `exemplar_indices_`). The `n === 1` early-return path still disposes `D_tensor` in `finally`. The precomputed path uses `matrix.clone()` (caller-owned) and `tf.tensor2d(rows)` (both disposed via `D_tensor` in `finally`).

Six `tf.memory()`-based tests were added to `src/memory_regression.test.ts`:

- Array input `fit()` + `dispose()` — no net tensor growth
- Tensor input `fit()` + `dispose()` — no net tensor growth
- Re-fitting (two sequential `fit()` calls) — no net tensor growth
- `n === 1` graceful noise path — `D_tensor` (1×1) still disposed via `finally`
- Precomputed metric (`number[][]`) — tensor upload and clone disposed correctly
- Large-n (n=500) — validates no argument-spread `RangeError` arises in the JS tail at scale

All 24 memory regression tests pass in 3.6 s. The HDBSCAN parity suite (58 tests) is unchanged.

**Acceptance criteria addressed:**

- **AC#1** — `tf.memory().numTensors` before === after for array-input and re-fit cases; `tf.tidy` and `finally` cover all intermediates.
- **AC#2** — `dispose()` sets all three attributes to `null`; confirmed by the re-fit test (no leaked instance tensors survive between fits).
- **AC#3** — `n === 1` and precomputed tests pass with no tensor leak; empty-input throws before `dispose()` (unchanged guard in `distance_matrix`).
- **AC#4** — n=500 fit completes in ~18 ms with no leak and no `RangeError`.

Original implementation notes from task creation:

Reuse the task-35 memory-regression pattern (`src/memory_regression.test.ts`). The single-readback design from 54.5 keeps the tensor lifetime short and bounded; the main audit targets are the input upload and the `tf.topk` outputs (indices tensor is unused and must be disposed). Keep the existing state-preservation invariant from task-52.5 intact.
