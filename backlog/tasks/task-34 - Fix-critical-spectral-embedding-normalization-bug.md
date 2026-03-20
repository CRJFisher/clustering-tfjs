---
id: task-34
title: Fix critical spectral embedding normalization bug
status: To Do
assignee: []
created_date: '2026-03-20'
labels: []
dependencies: []
---

## Description

The spectral embedding divides eigenvectors by D (degrees) instead of D^(1/2), which is mathematically incorrect. In spectral.ts lines 299-306, the code computes degrees = pow(sqrtDegrees, -2) giving D, then divides eigenvectors by D. The correct sklearn behavior divides by D^(1/2). This same bug appears in three places: fit(), fitWithIntermediateSteps(), and computeEmbeddingFromAffinity(). Additionally, SpectralClusteringConsensus uses a completely different (stale) diffusion map scaling approach that was explicitly removed from the main class. This distorts the embedding geometry and degrades clustering quality for harder problems.

## Acceptance Criteria

- [ ] Eigenvector normalization uses D^(1/2) not D in all three spectral embedding paths
- [ ] SpectralClusteringConsensus uses same normalization as main SpectralClustering class
- [ ] Spectral reference tests still pass with ARI >= 0.95
- [ ] Unit test added that verifies embedding values match sklearn numerically
