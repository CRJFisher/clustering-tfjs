---
id: TASK-52.10
title: >-
  Fix silhouette_score returning 0 silently for all-noise input instead of
  throwing
status: To Do
assignee: []
created_date: '2026-06-10 08:56'
labels:
  - bug
  - plausible
dependencies: []
references:
  - 'src/validation/silhouette.ts:152'
parent_task_id: TASK-52
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

silhouette_score previously documented that it throws when k≤1. The new noise-aware path returns 0 silently when all labels are -1 (all-noise input). A caller that passes all-noise labels — e.g. when HDBSCAN finds no clusters — sees a score of 0 and cannot distinguish a legitimately bad clustering from completely degenerate output, silently passing downstream validation gates.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [ ] #1 silhouette_score throws a descriptive error when all points are noise (all labels -1), matching the k≤1 contract
- [ ] #2 The docstring and/or type signature is updated to reflect the current behaviour contract
- [ ] #3 A degenerate-input test suite covers: all-noise labels (all -1), single-cluster labels, two-point clusters, and n_samples=0 — asserting that each either throws a descriptive error or returns the correct documented value (degenerate-input contract test)
<!-- AC:END -->
