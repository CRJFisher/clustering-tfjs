---
id: TASK-49.1
title: >-
  Density-clustering graph primitives: minimum spanning tree, k-distance, mutual
  reachability
status: Done
assignee: []
created_date: '2026-06-05'
updated_date: '2026-06-06 12:22'
labels: []
dependencies: []
parent_task_id: TASK-49
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

Density-based clustering builds its hierarchy from three graph and density primitives over a points-or-distance matrix: a minimum spanning tree of the mutual-reachability graph, a per-point core (k-distance) vector, and a mutual-reachability distance matrix. These primitives are reusable, independently verifiable building blocks that any density estimator consumes, so they live as standalone, unit-tested utilities in the graph and distance domains rather than buried inside a single estimator.

The k-nearest-neighbor affinity builder runs a single scan that yields, for every point, its neighbor indices and per-neighbor distances. Those indices and distances are exactly what the k-distance and mutual-reachability primitives need, so the affinity builder exposes them directly and the primitives consume them instead of recomputing distances.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [x] #1 `minimum_spanning_tree` in `src/graph/minimum_spanning_tree.ts` builds a Prim minimum spanning tree over a dense distance matrix and its edge set matches reference fixtures in `__fixtures__/density/`
- [x] #2 `kdistance` in `src/distance/kdistance.ts` returns a per-point core-distance vector equal to each point's k-th nearest-neighbor distance from the k-NN scan
- [x] #3 `mutual_reachability` in `src/graph/mutual_reachability.ts` returns a matrix where entry (i, j) equals `max(core_distance_i, core_distance_j, distance_i_j)`
- [x] #4 `compute_knn_affinity` in `src/graph/affinity.ts` returns `neighbor_indices` and `neighbor_distances` alongside the affinity matrix, produced by the single existing k-NN scan with no duplicated distance computation
- [x] #5 every caller of `compute_knn_affinity` consumes the single return shape with `neighbor_indices` and `neighbor_distances`, including the `compute_affinity_matrix` dispatcher in `src/graph/affinity.ts`, the SpectralClustering affinity path in `src/clustering/spectral.ts`, and the colocated affinity tests; no overload, wrapper, alias, or compatibility shim is introduced
- [x] #6 graph traversal for the minimum spanning tree runs in pure JavaScript over plain arrays, with tensors used only for the vectorized distance and k-NN computations
- [x] #7 all intermediate tensors created by the new primitives and by the affinity builder are disposed, with no net tensor leak across calls
- [x] #8 no `as any`, `as unknown`, or `as never` assertions appear in the new or modified code
- [x] #9 colocated tests `minimum_spanning_tree.test.ts`, `kdistance.test.ts`, and `mutual_reachability.test.ts` cover the primitives and pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->

Added minimum_spanning_tree (Prim, scipy parity), kdistance, mutual_reachability; compute_knn_affinity now returns {affinity, neighbor_indices, neighbor_distances} with all callers (compute_affinity_matrix, spectral, tests) updated. Fixtures via generate_density.py; colocated tests pass.

<!-- SECTION:NOTES:END -->
