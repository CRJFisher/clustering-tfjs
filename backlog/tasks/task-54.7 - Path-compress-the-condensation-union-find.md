---
id: TASK-54.7
title: Path-compress the condensation union-find
status: To Do
assignee: []
created_date: '2026-06-24'
updated_date: '2026-06-24'
labels:
  - hdbscan
  - graph
  - performance
dependencies:
  - task-54.1
parent_task_id: TASK-54
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

The single-linkage hierarchy in `src/graph/condensation_tree.ts` (`build_single_linkage`) uses an un-path-compressed union-find: `find()` walks `while (parent[root] !== -1) root = parent[root]` without compressing the traversed path, giving worst-case `O(n²)` lookup cost on degenerate chains. Add path compression so each `find()` points visited nodes directly at the root.

This is the one pure-JS performance win that survives the front-half migration — it accelerates the always-JS sequential tail and is independent of the tfjs stages, so it can be developed in parallel. It is strictly behavior-preserving: path compression changes only lookup cost, never the returned root, the merge order, or the resulting hierarchy.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [ ] #1 `find()` in `build_single_linkage` compresses the traversed path while returning identical roots to the un-compressed version
- [ ] #2 Merge-order semantics are untouched: edges are still sorted by weight only, and unions assign labels in the same order; `build_single_linkage` produces byte-identical hierarchy rows to the baseline
- [ ] #3 `condensation_tree.test.ts` and `hdbscan.test.ts` pass with labels/probabilities bit-identical to the 54.1 oracle (this change introduces no float32 and must not shift any value)

<!-- AC:END -->

## Implementation Notes

Behavior-preserving and float64-only, so it is held to the _bit-identical_ bar (not the relaxed float32 tolerances). The optional broader condensation-tree dedup (unifying the two BFS helpers, sharing the births map) is explicitly out of scope per YAGNI — it carries load-bearing tie-ordering risk for no measured perf gain; revisit only if it falls out cleanly here.
