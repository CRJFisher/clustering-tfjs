---
id: TASK-49.4
title: HDBSCAN condensed tree and excess-of-mass cluster selection
status: Done
assignee: []
created_date: '2026-06-05'
updated_date: '2026-06-06 12:22'
labels: []
dependencies:
  - task-49.1
parent_task_id: TASK-49
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

HDBSCAN derives a flat clustering from a hierarchy of density-connected components. Two pieces form the core of that derivation: a condensed tree that collapses the single-linkage hierarchy over the minimum spanning tree into a compact set of candidate clusters annotated with birth/death density levels and population, and a stability-based selection that picks the final flat clustering from those candidates. This module provides both as a standalone unit that operates on graph structures and emits cluster assignments, independent of any estimator wrapper. Isolating the density hierarchy and selection logic lets them be validated against sklearn reference values on their own before an estimator consumes them.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [x] #1 `build_condensation_tree` consumes the minimum spanning tree produced in `src/graph/minimum_spanning_tree.ts` together with mutual-reachability edge ordering and produces a condensed tree whose nodes carry birth lambda, death lambda, and population (cluster size) fields
- [x] #2 `excess_of_mass` stability selection extracts a flat clustering from the condensed tree whose labels match sklearn HDBSCAN on golden small inputs loaded from `__fixtures__/hdbscan/` (repo-root fixtures consumed via `process.cwd()` in `src/graph/condensation_tree.test.ts`)
- [x] #3 `cluster_selection_method` supports both `'eom'` and `'leaf'`, and each produces the sklearn-matching labels for its mode on the golden fixtures
- [x] #4 `min_cluster_size` is honored when condensing the hierarchy: components smaller than `min_cluster_size` are treated as points falling out of their parent rather than as distinct clusters
- [x] #5 degenerate inputs (all points equidistant, and inputs that resolve to a single cluster) produce finite labels and stability values with no NaN
- [x] #6 `src/graph/condensation_tree.ts` exposes only snake_case functions and PascalCase types, threads no compatibility shims or aliases, performs no unsafe type assertions (no `as any` / `as unknown` / `as never`), and is covered by colocated `src/graph/condensation_tree.test.ts`
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->

condensation*tree.ts: build_single_linkage, condense_hierarchy/build_condensation_tree, compute_stability, excess_of_mass (eom+leaf+epsilon), extract_labels. Validated bit-exact against sklearn's \_single_linkage_tree* hierarchy; degenerate inputs finite.

<!-- SECTION:NOTES:END -->
