---
id: TASK-54.5
title: Compute mutual-reachability on-tensor and define the single readback boundary
status: To Do
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

- [ ] #1 Mutual-reachability is computed with broadcast `tf.maximum` over the core-distance vector (as column and row) and the distance tensor; the standalone JS `mutual_reachability` double loop is no longer on the HDBSCAN production path
- [ ] #2 The distance → core → mutual-reachability stages execute within a single `tf.tidy` (or explicit dispose) block with no leaked intermediates
- [ ] #3 Exactly one GPU→CPU readback occurs, producing a flat row-major `Float32Array` (length `n*n`) consumed by the MST stage; no earlier stage reads back to JS
- [ ] #4 `hdbscan.test.ts` passes under the 54.2 tolerances with labels exactly matching the oracle

<!-- AC:END -->

## Implementation Notes

The flat `Float32Array` readback is deliberate: `minimum_spanning_tree` already supports a flat typed-array input with an `n` argument and a cache-friendly `at(i,j) = arr[i*n+j]` accessor (54.6 wires this up). Prefer `.data()` (async, flat) over `.array()` (nested) to avoid rebuilding a nested `number[][]`. Keep the precomputed-input path flowing through the same on-tensor stages (upload → topk → max → readback) so there is a single code path.
