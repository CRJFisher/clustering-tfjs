---
id: TASK-55.7
title: Interactive scikit-learn toy-dataset grid across all five algorithms
status: To Do
assignee: []
created_date: '2026-06-26'
updated_date: '2026-06-26'
labels:
  - demo
  - ux
dependencies:
  - task-55.4
parent_task_id: TASK-55
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

Recreate the iconic scikit-learn "Comparing clustering algorithms on toy datasets" image — but live and interactive. Rows = moons / circles / blobs / anisotropic / no-structure; columns = K-Means, Spectral, Agglomerative, HDBSCAN, SOM; each cell a live canvas scatter colored by cluster, computed via the real library in a worker. This section sells trust/parity: it signals "this is the sklearn clustering you already know, in your browser."

Compute runs off the main thread and re-renders without freezing the page. Curate datasets/params where float32 parity with sklearn holds; where it does not (the known HDBSCAN/Spectral float32 probability drift), annotate the expected difference rather than hiding it — the parity claim must not be undercut. KMeans may show animated convergence here (it is animation eye-candy, not a speedup claim).

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [ ] #1 All five algorithms produce live cluster colorings across all five dataset rows
- [ ] #2 Compute runs off the main thread and re-renders without freezing the page
- [ ] #3 Datasets/params are curated so float32 labels match the sklearn-parity story, or differences are annotated
- [ ] #4 The grid is visually legible with high-contrast (colorblind-safe) cluster colors at the hero framing
- [ ] #5 ESLint passes before commit

<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->

Open question for delivery: identify the exact dataset+param combinations that keep HDBSCAN/Spectral labels matching sklearn under float32. Consider OffscreenCanvas in a render worker only if main-thread 2D-canvas rendering of the many simultaneous cells proves insufficient at the capped point counts.

<!-- SECTION:NOTES:END -->
