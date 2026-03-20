---
id: TASK-34
title: Fix critical spectral embedding normalization bug
status: Done
assignee: []
created_date: '2026-03-20'
updated_date: '2026-03-20 15:24'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The spectral embedding divides eigenvectors by D (degrees) instead of D^(1/2), which is mathematically incorrect. In spectral.ts lines 299-306, the code computes degrees = pow(sqrtDegrees, -2) giving D, then divides eigenvectors by D. The correct sklearn behavior divides by D^(1/2). This same bug appears in three places: fit(), fitWithIntermediateSteps(), and computeEmbeddingFromAffinity(). Additionally, SpectralClusteringConsensus uses a completely different (stale) diffusion map scaling approach that was explicitly removed from the main class. This distorts the embedding geometry and degrades clustering quality for harder problems.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Eigenvector normalization uses D^(1/2) not D in all three spectral embedding paths
- [x] #2 SpectralClusteringConsensus uses same normalization as main SpectralClustering class
- [x] #3 Spectral reference tests still pass with ARI >= 0.95
- [x] #4 Unit test added that verifies embedding values match sklearn numerically
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Fix tf.pow(sqrtDegrees, -2) to tf.pow(sqrtDegrees, -1) in all 4 normalization sites in spectral.ts
2. Replace diffusion map scaling in SpectralClusteringConsensus with degree normalization
3. Generate sklearn embedding fixtures with numerical embedding values
4. Add Gram matrix and column-wise numerical embedding verification tests
5. Add SpectralClusteringConsensus test
6. Fix tensor leak in fitWithIntermediateSteps result preparation
7. Update stale diffusion map comments
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed the spectral embedding normalization bug where eigenvectors were divided by D (full degree) instead of D^{1/2} (square root of degree). The root cause was tf.pow(sqrtDegrees, -2) computing D instead of tf.pow(sqrtDegrees, -1) computing D^{1/2}, given that sqrtDegrees from normalised_laplacian is D^{-1/2}.

Fixed in 4 locations in spectral.ts (fit, fitWithIntermediateSteps, intermediate result, computeEmbeddingFromAffinity) and rewrote SpectralClusteringConsensus to use degree normalization instead of stale diffusion map scaling.

Added numerical embedding verification tests using sklearn fixtures: Gram matrix comparison for degenerate eigenspaces (blobs) and column-wise cosine similarity for distinct eigenvalues (moons). Added SpectralClusteringConsensus test. Fixed tensor leak in fitWithIntermediateSteps result preparation.

43/44 spectral tests pass. The sole failure (circles_n3_rbf ARI=0.57) is pre-existing on main.

Modified files: src/clustering/spectral.ts, src/clustering/spectral_consensus.ts, src/utils/smallest_eigenvectors_with_values.ts, src/utils/platform.ts
Added files: test/clustering/spectral_embedding_numerical.test.ts, test/clustering/spectral_consensus.test.ts, test/fixtures/spectral/embedding_blobs_n3_rbf.json, test/fixtures/spectral/embedding_moons_n2_rbf.json, tools/sklearn_fixtures/generate_spectral_embedding.py
<!-- SECTION:NOTES:END -->
