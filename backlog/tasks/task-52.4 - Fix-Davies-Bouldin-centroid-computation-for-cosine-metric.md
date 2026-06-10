---
id: TASK-52.4
title: Fix Davies-Bouldin centroid computation for cosine metric
status: To Do
assignee: []
created_date: '2026-06-10 08:55'
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

- [ ] #1 When metric='cosine', cluster centroids are computed as L2-normalised means (spherical centroid) rather than raw Euclidean means
- [ ] #2 The DB score for a well-separated cosine cluster is lower than for a poorly-separated one
- [ ] #3 Existing euclidean-metric tests are unaffected
- [ ] #4 A test uses clusters of antipodal unit vectors (where the Euclidean centroid is near the origin) and asserts the resulting DB score is a sensible finite value, not the near-1.0 result produced by dividing by the eps guard on a zero-norm centroid (metric-specific edge-case test)
<!-- AC:END -->
