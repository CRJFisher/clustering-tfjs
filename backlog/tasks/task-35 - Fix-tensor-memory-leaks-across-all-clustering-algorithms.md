---
id: task-35
title: Fix tensor memory leaks across all clustering algorithms
status: To Do
assignee: []
created_date: '2026-03-20'
labels: []
dependencies: []
---

## Description

Multiple reviewers identified systematic tensor memory leaks throughout the codebase. KMeans leaks 6000+ tensors per fit() call from undisposed argMin/min intermediates and centroid shift chains. SpectralClustering leaks tf.cast results when input is a tensor, plus sum().data() and tf.pow intermediates. SOM visualization leaks U-matrix tensors. Validation functions redundantly dispose inside tf.tidy. KMeans has no dispose() method at all, so centroids_ leak permanently. findOptimalClusters never disposes algorithm instances between k evaluations.

## Acceptance Criteria

- [ ] KMeans.fit wraps argMin/min/centroid-shift ops in tf.tidy or explicitly disposes intermediates
- [ ] KMeans class has a dispose() method that releases centroids_ tensor
- [ ] SpectralClustering.fit disposes tf.cast result when input is a tensor
- [ ] SpectralClustering.fitWithIntermediateSteps disposes tf.pow intermediate
- [ ] Validation functions use consistent tidy-or-manual-dispose pattern (not both)
- [ ] findOptimalClusters disposes clustering instances after each k evaluation
- [ ] SOM exportForVisualization disposes U-matrix tensor
- [ ] Memory regression test added that asserts tensor count before/after fit+dispose
