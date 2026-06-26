---
id: TASK-54.6
title: Feed the flat readback buffer into the JS minimum spanning tree
status: Done
assignee: []
created_date: '2026-06-24'
updated_date: '2026-06-24'
labels:
  - hdbscan
  - graph
dependencies:
  - task-54.5
parent_task_id: TASK-54
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

Wire the flat `Float32Array` mutual-reachability buffer from 54.5 into the sequential tail. Prim's minimum spanning tree stays float64 JS (it must — loop-carried frontier dependency, per-iteration scalar readback for the disconnect guard and edge recording), but it consumes the contiguous buffer directly through its existing flat-array path rather than a rebuilt nested `number[][]`.

`minimum_spanning_tree` already accepts `Float64Array | number[][]` with an `n` argument and a row-major `at(i,j) = arr[i*n+j]` accessor. This subtask confirms/extends that path to accept the `Float32Array` readback and routes HDBSCAN through it. No algorithmic change — the tail is identical; only its input representation becomes the flat buffer.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [ ] #1 `minimum_spanning_tree` consumes the flat typed-array buffer (with `n`) produced by 54.5; HDBSCAN no longer materialises a nested `number[][]` mutual-reachability matrix for the MST
- [ ] #2 No tensor op is introduced into `minimum_spanning_tree` or any tail stage; Prim's algorithm and its edge ordering are unchanged
- [ ] #3 `hdbscan.test.ts` and `minimum_spanning_tree.test.ts` pass under the 54.2 tolerances with labels exactly matching the oracle

<!-- AC:END -->

## Implementation Notes

## High-level summary

`minimum_spanning_tree` accepts the flat `Float32Array` produced by the single tensor readback, and HDBSCAN routes the mutual-reachability matrix through it directly — no nested `number[][]` reconstruction at the MST boundary.

The function signature was widened from `Float64Array | number[][]` to `Float32Array | Float64Array | number[][]`. The flat-path discriminator changed from `instanceof Float64Array` to `ArrayBuffer.isView(distance_matrix)`, which correctly catches both typed-array forms and reads in the positive direction. The existing row-major `at(i, j) = arr[i * n + j]` accessor is unchanged; the Prim's algorithm and edge ordering are untouched. HDBSCAN's call site passes `(mreach_flat, n)` with explicit `n` — the inference-from-sqrt fallback is not used on the production path.

This subtask was completed as part of task-54.5: the HDBSCAN `fit` method's single `.data()` readback, the type widening, and the call-site update landed together in that commit. This confirmation pass verified all three acceptance criteria against the live code and tests (68 pass), and a 10-lens review produced five minor fixes:

- `mutual_reachability.ts` module JSDoc now states it is a test-only reference oracle; production mutual reachability runs on-tensor in HDBSCAN.
- HDBSCAN class JSDoc now names the single `.data()` readback as the front-half/tail boundary.
- `minimum_spanning_tree` function JSDoc now documents both input forms, notes that the flat path is what HDBSCAN uses, and flags that explicit `n` is required for non-square buffers.
- The `best_weight[0] = 0` Prim's seed carries a brief why-comment.

**Acceptance criteria addressed:**

- **AC#1** — `minimum_spanning_tree(mreach_flat, n)` at `hdbscan.ts:254`; the `mutual_reachability` import is removed; no `number[][]` matrix is materialised for the MST.
- **AC#2** — `minimum_spanning_tree.ts` is pure JS Prim's; no `tf.*` op anywhere in the tail.
- **AC#3** — 68 tests pass under the task-54.2 tolerances; labels match the oracle exactly.

The accessor and flat path already exist (`src/graph/minimum_spanning_tree.ts`); the likely change is only widening the accepted type to include `Float32Array` and updating HDBSCAN's call site. The cache-friendly contiguous read in the relaxation pass is a free secondary win over nested-array pointer chasing.
