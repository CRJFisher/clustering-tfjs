---
id: TASK-49
title: Topic Detection and Tracking (TDT) upgrade
status: Done
assignee: []
created_date: '2026-06-05'
updated_date: '2026-06-06 12:23'
labels:
  - tdt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Equip the library with the capability bundle a browsing-timeline consumer needs for Topic Detection and Tracking: discover variable-density topics over embedded browsing events without a preset cluster count and flag incidental pages as noise (HDBSCAN), measure topic similarity with cosine geometry, expose a uniform representative vector for every estimator, and match topics across consecutive time windows into `emerge / persist / merge / split / die` lifelines.

This is the umbrella task for that effort. The work is decomposed into subtasks `task-49.1`–`task-49.10`, sequenced by their dependencies: shared density/graph primitives and the `-1` noise contract first, then cosine, then HDBSCAN, then representation accessors, then cross-window tracking, with a public PCA estimator and KMeans serialization as demand-driven follow-ups. The full design, sequencing rationale, fixture strategy, and hard-parts analysis live in `backlog/docs/tdt-upgrade-plan.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 HDBSCAN discovers variable-density clusters and emits `-1` noise, matching scikit-learn parity fixtures (`task-49.5`)
- [x] #2 Cosine is selectable as a first-class metric/affinity across KMeans, Spectral, Agglomerative, and the relevant validation metrics (`task-49.3`)
- [x] #3 Every estimator exposes a representative vector through one uniform `ClusterRepresentations` surface (`task-49.6`, `task-49.7`)
- [x] #4 `track_clusters` matches clusters across consecutive snapshots and emits transitions with stable lifeline IDs (`task-49.8`)
- [x] #5 Validation metrics are noise-aware and the cluster-label contract is recorded as a decision (`task-49.2`)
- [x] #6 All subtasks `task-49.1`–`task-49.10` are complete with their own acceptance criteria met, or explicitly deferred where marked demand-driven
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
All subtasks 49.1-49.10 complete. TDT bundle shipped: density primitives, -1 noise contract + noise-aware metrics, cosine first-class, HDBSCAN (condensed tree + EOM), uniform ClusterRepresentations (centroids/medoids/exemplars), KMeans predict + serialization, cross-window track_clusters, public PCA. sklearn parity fixtures throughout.
<!-- SECTION:NOTES:END -->
