---
id: TASK-52.7
title: Fix SpectralClustering dispose() not clearing medoid_indices_
status: To Do
assignee: []
created_date: '2026-06-10 08:56'
labels:
  - bug
  - plausible
dependencies: []
references:
  - 'src/clustering/spectral.ts:127'
parent_task_id: TASK-52
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

SpectralClustering.dispose() resets labels* and affinity matrices but does not clear medoid_indices*. A consumer that calls fit(data1), compute_medoids(), then fit(data2) will read stale medoid indices from the first fit until compute_medoids() is explicitly called again — the indices are into data1's label space but are silently presented as valid for data2.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [ ] #1 dispose() sets medoid*indices* to null alongside the other per-fit state
- [ ] #2 A re-fit after dispose() leaves medoid*indices* null until compute_medoids() is called again
- [ ] #3 A test verifies that all per-fit output fields (labels*, affinity_matrix*, medoid*indices*) are null immediately after fit(data2) completes and before any accessor is called (complete per-fit state-reset test)
<!-- AC:END -->
