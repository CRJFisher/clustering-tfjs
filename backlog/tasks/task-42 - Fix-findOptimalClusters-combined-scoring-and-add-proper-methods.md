---
id: task-42
title: Fix findOptimalClusters combined scoring and add proper methods
status: To Do
assignee: []
created_date: '2026-03-20'
labels: []
dependencies: []
---

## Description

The findOptimalClusters default combined score adds raw unnormalized metrics (silhouette [-1,1] + calinskiHarabasz [0,thousands] - daviesBouldin [0,infinity]), making Calinski-Harabasz completely dominate. The function also lacks elbow method, gap statistic, and proper silhouette-only method. Validation metrics have additional issues: silhouetteScore NaN when both a and b are 0, daviesBouldin returns Infinity for coincident centroids, no validation that labels length matches data length, no noise label (-1) handling, no per-sample silhouette scores exposed.

## Acceptance Criteria

- [ ] Combined score uses normalized metrics so all three contribute meaningfully
- [ ] Elbow method implemented using WSS/inertia curve knee detection
- [ ] Silhouette-only method implemented
- [ ] silhouetteScore returns 0 (not NaN) when a==0 and b==0
- [ ] daviesBouldin handles coincident centroids matching sklearn behavior
- [ ] Labels length validated against data rows in all validation functions
- [ ] Per-sample silhouette scores exposed as silhouetteSamples function
- [ ] Missing metrics: ARI and NMI added for supervised evaluation
