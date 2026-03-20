---
id: TASK-35
title: Fix tensor memory leaks across all clustering algorithms
status: Done
assignee:
  - '@claude'
created_date: '2026-03-20'
updated_date: '2026-03-20 13:52'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Multiple reviewers identified systematic tensor memory leaks throughout the codebase. KMeans leaks 6000+ tensors per fit() call from undisposed argMin/min intermediates and centroid shift chains. SpectralClustering leaks tf.cast results when input is a tensor, plus sum().data() and tf.pow intermediates. SOM visualization leaks U-matrix tensors. Validation functions redundantly dispose inside tf.tidy. KMeans has no dispose() method at all, so centroids_ leak permanently. findOptimalClusters never disposes algorithm instances between k evaluations.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 KMeans.fit wraps argMin/min/centroid-shift ops in tf.tidy or explicitly disposes intermediates
- [x] #2 KMeans class has a dispose() method that releases centroids_ tensor
- [x] #3 SpectralClustering.fit disposes tf.cast result when input is a tensor
- [x] #4 SpectralClustering.fitWithIntermediateSteps disposes tf.pow intermediate
- [x] #5 Validation functions use consistent tidy-or-manual-dispose pattern (not both)
- [x] #6 findOptimalClusters disposes clustering instances after each k evaluation
- [x] #7 SOM exportForVisualization disposes U-matrix tensor
- [x] #8 Memory regression test added that asserts tensor count before/after fit+dispose
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add dispose() to KMeans (AC#2) - prerequisite for other fixes
2. Fix KMeans.fit() tensor leaks: argMin/min, slice, centroid shift chain (AC#1)
3. Fix SpectralClustering tf.cast leak by always disposing Xtensor (AC#3)
4. Fix tf.pow intermediate in fitWithIntermediateSteps (AC#4)
5. Remove redundant manual disposes inside tf.tidy in validation functions (AC#5)
6. Dispose clustering instances in findOptimalClusters loop (AC#6)
7. Dispose U-matrix tensor in exportForVisualization via try/finally (AC#7)
8. Create memory regression test suite (AC#8)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented all 8 acceptance criteria. Fixed tensor leaks in KMeans, SpectralClustering, validation functions, findOptimalClusters, and SOM visualization. Added 15 memory regression tests.
<!-- SECTION:NOTES:END -->
