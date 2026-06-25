---
id: TASK-54.6
title: Feed the flat readback buffer into the JS minimum spanning tree
status: To Do
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

The accessor and flat path already exist (`src/graph/minimum_spanning_tree.ts`); the likely change is only widening the accepted type to include `Float32Array` and updating HDBSCAN's call site. The cache-friendly contiguous read in the relaxation pass is a free secondary win over nested-array pointer chasing.
