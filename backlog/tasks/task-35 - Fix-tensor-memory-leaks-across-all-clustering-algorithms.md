---
id: TASK-35
title: Fix tensor memory leaks across all clustering algorithms
status: In Progress
assignee:
  - '@claude'
created_date: '2026-03-20'
updated_date: '2026-03-20 12:00'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Multiple reviewers identified systematic tensor memory leaks throughout the codebase. KMeans leaks 6000+ tensors per fit() call from undisposed argMin/min intermediates and centroid shift chains. SpectralClustering leaks tf.cast results when input is a tensor, plus sum().data() and tf.pow intermediates. SOM visualization leaks U-matrix tensors. Validation functions redundantly dispose inside tf.tidy. KMeans has no dispose() method at all, so centroids_ leak permanently. findOptimalClusters never disposes algorithm instances between k evaluations.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 KMeans.fit wraps argMin/min/centroid-shift ops in tf.tidy or explicitly disposes intermediates
- [ ] #2 KMeans class has a dispose() method that releases centroids_ tensor
- [ ] #3 SpectralClustering.fit disposes tf.cast result when input is a tensor
- [ ] #4 SpectralClustering.fitWithIntermediateSteps disposes tf.pow intermediate
- [ ] #5 Validation functions use consistent tidy-or-manual-dispose pattern (not both)
- [ ] #6 findOptimalClusters disposes clustering instances after each k evaluation
- [ ] #7 SOM exportForVisualization disposes U-matrix tensor
- [ ] #8 Memory regression test added that asserts tensor count before/after fit+dispose
<!-- AC:END -->
