---
id: TASK-49
title: Density, cosine, and representation clustering upgrade
status: Done
assignee: []
created_date: '2026-06-05'
updated_date: '2026-06-06 12:23'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

Extend the library with a bundle of density-based and geometry-aware clustering capabilities: discover variable-density clusters without a preset cluster count and flag outliers as noise (HDBSCAN), measure similarity with cosine geometry, and expose a uniform representative vector for every estimator.

This is the umbrella task for that effort. The work is decomposed into subtasks `task-49.1`–`task-49.10`, sequenced by their dependencies: shared density/graph primitives and the `-1` noise contract first, then cosine, then HDBSCAN, then representation accessors, with a public PCA estimator and KMeans serialization as demand-driven follow-ups.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [x] #1 HDBSCAN discovers variable-density clusters and emits `-1` noise, matching scikit-learn parity fixtures (`task-49.5`)
- [x] #2 Cosine is selectable as a first-class metric/affinity across KMeans, Spectral, Agglomerative, and the relevant validation metrics (`task-49.3`)
- [x] #3 Every estimator exposes a representative vector through one uniform `ClusterRepresentations` surface (`task-49.6`, `task-49.7`)
- [x] #4 Validation metrics are noise-aware and the cluster-label contract is recorded as a decision (`task-49.2`)
- [x] #5 All subtasks `task-49.1`–`task-49.10` are complete with their own acceptance criteria met, or explicitly deferred where marked demand-driven
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->

All subtasks 49.1-49.10 complete: density primitives, -1 noise contract + noise-aware metrics, cosine first-class, HDBSCAN (condensed tree + EOM), uniform ClusterRepresentations (centroids/medoids/exemplars), KMeans predict + serialization, public PCA. sklearn parity fixtures throughout.

<!-- SECTION:NOTES:END -->
