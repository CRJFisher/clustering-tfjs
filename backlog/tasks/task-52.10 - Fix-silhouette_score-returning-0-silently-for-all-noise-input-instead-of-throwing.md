---
id: TASK-52.10
title: >-
  Fix silhouette_score returning 0 silently for all-noise input instead of
  throwing
status: Done
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

## Implementation Notes

### High-level summary

`silhouette_score` and `silhouette_samples` now throw `"Silhouette score requires at least 2 clusters; all labels are noise"` when all input labels are `-1`. The prior behavior returned `0` silently by routing through the `had_noise` branch of a unified `k <= 1` guard, making degenerate all-noise output indistinguishable from a genuine boundary score. The fix splits that guard: `k === 0` (all noise, zero valid clusters) throws unconditionally; `k === 1` with noise (one valid cluster remains after filtering) continues to return a defined zero. `silhouette_score_subset` receives the same fix. The previously dead `samples.length === 0` guard in `silhouette_score` is removed.

**Acceptance criteria addressed:**

- AC#1: All three public functions throw with a descriptive message for all-noise input (`src/validation/silhouette.ts` L60–65, L192–197).
- AC#2: Docstrings on `silhouette_samples`, `silhouette_score`, and `silhouette_score_subset` are updated to document the all-noise-throws / one-cluster-plus-noise-returns-0 / single-cluster-no-noise-throws contract. `docs/API.md` updated to match.
- AC#3: Degenerate-input contract test suite added (`silhouette.test.ts` "Silhouette – degenerate-input contract (AC#3)"): all-noise throws, single-cluster throws, two-point two-cluster returns finite score, `n_samples=0` throws on length mismatch, single-cluster-plus-noise returns 0.

**Key invariant preserved:** `k === 1` with `had_noise` (one valid cluster + noise points) remains non-throwing and returns a defined 0, consistent with the DBSCAN/HDBSCAN noise convention.

**Review findings applied:** `silhouette_score_subset` JSDoc was missing `@throws`, `@param metric`, and any noise-contract description (3 independent review lenses). `silhouette_score @throws` wording was tightened to name the two distinct throwing conditions without implying one-cluster-plus-noise throws.
