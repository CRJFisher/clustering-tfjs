---
id: TASK-54.5
title: Compute mutual-reachability on-tensor and define the single readback boundary
status: Done
assignee: []
created_date: '2026-06-24'
updated_date: '2026-06-24'
labels:
  - hdbscan
  - tfjs
  - performance
dependencies:
  - task-54.4
parent_task_id: TASK-54
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

Compute the mutual-reachability matrix `M[i][j] = max(core_i, core_j, d_ij)` on the backend as a broadcast `tf.maximum(tf.maximum(core_col, core_row), D)`, fusing the front-half (distance from 54.3, core distances from 54.4, this max) into a single `tf.tidy` block. Then perform the **one and only** GPU→CPU readback of the whole pipeline, materialising the mutual-reachability matrix as a flat row-major `Float32Array` to feed the JS minimum spanning tree.

This subtask is where the front-half performance win is realised: everything from input upload through mutual-reachability stays on-tensor, and a single `.data()` readback hands a contiguous buffer to the sequential tail — no per-stage round-trips.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [x] #1 Mutual-reachability is computed with broadcast `tf.maximum` over the core-distance vector (as column and row) and the distance tensor; the standalone JS `mutual_reachability` double loop is no longer on the HDBSCAN production path
- [x] #2 The distance → core → mutual-reachability stages execute within a single `tf.tidy` (or explicit dispose) block with no leaked intermediates
- [x] #3 Exactly one GPU→CPU readback occurs, producing a flat row-major `Float32Array` (length `n*n`) consumed by the MST stage; no earlier stage reads back to JS
- [x] #4 `hdbscan.test.ts` passes under the 54.2 tolerances with labels exactly matching the oracle

<!-- AC:END -->

## Implementation Notes

## High-level summary

This subtask realises the front-half performance win: everything from input upload through mutual-reachability now stays on the TF.js backend, and a single `.data()` readback hands a flat `Float32Array` to the sequential MST tail — no per-stage round-trips.

The approach fuses all three front-half stages inside a single `tf.tidy` in `HDBSCAN.fit()`. `core_distances` (already on-tensor via `tf.topk`, from task-54.4) is called inside the outer tidy and returns a `Tensor1D`; the tidy then broadcasts it as `core.reshape([n, 1])` and `core.reshape([1, n])` over the distance tensor with nested `tf.maximum`, implementing `M[i,j] = max(core[i], core[j], D[i,j])` in three tensor ops with no intermediate readback. On exit the tidy disposes the core vector, its two reshaped views, and the intermediate maximum; `M_tensor` is the sole returned output. A single `await M_tensor.data()` produces the flat row-major `Float32Array` of length `n*n`; `M_tensor` is immediately disposed and `D_tensor` follows in `finally`.

The JS `mutual_reachability` double-loop is removed from the HDBSCAN production path; its module (`graph/mutual_reachability.ts`) is retained as a test-only reference oracle used by `condensation_tree.test.ts`. `minimum_spanning_tree` is widened to accept `Float32Array | Float64Array | number[][]` — the `is_flat` discriminator changes from `instanceof Float64Array` to `!(instanceof Array)`, and the existing flat-path accessor covers both typed-array types. The HDBSCAN call site passes `(mreach_flat, n)` explicitly.

**What to know:** The precomputed-metric path flows through the same single code path (`distance_matrix()` returns a `Tensor2D`; the fused tidy handles it identically). The `mutual_reachability.ts` module has zero production callers after this change; its final cleanup is gated on task-54.10 (benchmark-justified decision on whether a small-`n` JS fallback is retained). The tensor-leak regression test for HDBSCAN is tracked for task-54.8.

**Acceptance criteria addressed:**

- **AC#1** — Broadcast `tf.maximum(tf.maximum(core.reshape([n,1]), core.reshape([1,n])), D_tensor)` at `hdbscan.ts:240-246`. `mutual_reachability` import removed; no non-test production caller remains.
- **AC#2** — Distance, core, and mutual-reachability all execute inside one `tf.tidy`. `core_distances`'s own inner tidy handles its intermediates; the outer tidy handles the core vector and the two broadcast ops. `D_tensor` is externally owned and disposed in `finally`.
- **AC#3** — The only GPU→CPU transfer is `await M_tensor.data()` at `hdbscan.ts:249`, which returns a flat row-major `Float32Array` of length `n*n`. `minimum_spanning_tree(mreach_flat, n)` consumes it directly.
- **AC#4** — 68 tests pass (58 HDBSCAN parity suite + 10 MST unit tests including two new Float32Array path tests). Labels match the oracle exactly across all 33 fixtures under the task-54.2 tolerances.

**Review.** A 10-lens review (correctness ×4, completeness ×2, IA ×3, adversarial cold-read) found no correctness defects. Three fixes were applied: the class JSDoc now accurately states mutual reachability is a fused `tf.maximum` broadcast in `fit` rather than sourced from `graph/`; the comment above the `tf.tidy` block no longer claims "negated D, topk outputs" as its intermediates (those live in `core_distances`'s inner tidy); and a test for Float32Array without explicit `n` was added to cover the sqrt-inference path now documented in the MST JSDoc.

The original implementation notes below:

The flat `Float32Array` readback is deliberate: `minimum_spanning_tree` already supports a flat typed-array input with an `n` argument and a cache-friendly `at(i,j) = arr[i*n+j]` accessor (54.6 wires this up). Prefer `.data()` (async, flat) over `.array()` (nested) to avoid rebuilding a nested `number[][]`. Keep the precomputed-input path flowing through the same on-tensor stages (upload → topk → max → readback) so there is a single code path.
