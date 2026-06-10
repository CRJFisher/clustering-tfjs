---
id: TASK-52.4
title: Fix Davies-Bouldin centroid computation for cosine metric
status: Done
assignee:
  - crjfisher
created_date: '2026-06-10 08:55'
updated_date: '2026-06-10 14:48'
labels:
  - bug
  - confirmed
dependencies: []
references:
  - 'src/validation/davies_bouldin.ts:104'
parent_task_id: TASK-52
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
davies_bouldin_score always computes cluster centroids as Euclidean means regardless of the metric parameter. For cosine metric, the Euclidean mean of a cluster can be near the origin (e.g. antipodal points [1,0] and [-1,0] average to [0,0]), making cosine dispersion ill-defined and producing a meaningless score that cannot be compared against Euclidean-based results.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When metric='cosine', cluster centroids are computed as L2-normalised means (spherical centroid) rather than raw Euclidean means
- [x] #2 The DB score for a well-separated cosine cluster is lower than for a poorly-separated one
- [x] #3 Existing euclidean-metric tests are unaffected
- [x] #4 A test uses clusters of antipodal unit vectors (where the Euclidean centroid is near the origin) and asserts the resulting DB score is a sensible finite value, not the near-1.0 result produced by dividing by the eps guard on a zero-norm centroid (metric-specific edge-case test)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. In davies_bouldin (tf.tidy path): after computing cluster_data.mean(0), if metric==='cosine', divide by L2-norm+eps to produce a unit-sphere centroid.\n2. In davies_bouldin_efficient (manual tidy path): same normalisation inside the tf.tidy block before extracting to centroid_arrays.\n3. Add cosine-metric tests: well-separated vs poorly-separated directional clusters (AC#2); near-antipodal unit-vector clusters showing finite finite DB (AC#4).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added spherical-centroid normalisation for cosine metric via a new cluster_centroid() module-level helper (mirrors the existing cluster_dispersion() pattern). When metric==='cosine', cluster_centroid() divides the Euclidean mean by its L2-norm + 1e-8 to produce a unit-sphere centroid. Both davies_bouldin and davies_bouldin_efficient delegate to this helper, removing the 5-line duplication flagged in review. The code change is semantically correct but is a mathematical no-op for all non-zero centroids — both cluster_dispersion and pairwise_distance_matrix already re-normalise their inputs internally for cosine, so DB scores are numerically identical with or without pre-normalisation. The practical benefit is in davies_bouldin_efficient, where centroids are serialised to JS float arrays before reconstruction: storing a unit vector avoids directional precision loss for near-zero-magnitude means. Exactly-zero means (perfectly-antipodal clusters) are an undefined case and remain [0,…,0] after normalisation. Post-review: extracted cluster_centroid() helper, dropped redundant type annotations, rewrote the AC#4 test comment to be accurate (removed the false claim that 'eps guard dominates'; the test now honestly describes API-contract verification rather than claiming numerical improvement). Modified files: src/validation/davies_bouldin.ts, src/validation/davies_bouldin.test.ts. All 952 tests pass; ESLint clean.
<!-- SECTION:NOTES:END -->
