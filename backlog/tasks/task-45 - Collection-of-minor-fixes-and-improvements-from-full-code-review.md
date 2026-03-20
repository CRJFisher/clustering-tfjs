---
id: TASK-45
title: Collection of minor fixes and improvements from full code review
status: In Progress
assignee: []
created_date: '2026-03-20'
updated_date: '2026-03-20 21:26'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
This task collects smaller improvements identified during the comprehensive 15-agent code review that don't warrant individual tasks. These should be addressed opportunistically or as a cleanup pass.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 DOCUMENTATION: Update CHANGELOG.md for versions 0.2.0 through 0.4.0
- [ ] #2 DOCUMENTATION: Remove predict() from README common interface (only SOM has it)
- [ ] #3 DOCUMENTATION: Remove non-existent init parameter from KMeans docs in README and API.md
- [ ] #4 DOCUMENTATION: Add SOM to README API Reference section
- [ ] #5 DOCUMENTATION: Fix broken links to non-existent performance.md and migration-guide.md
- [ ] #6 DOCUMENTATION: Remove stale 'coming soon' and 'backend selection coming in future' comments from README
- [ ] #7 DOCUMENTATION: Update placeholder author in package.json
- [ ] #8 DOCUMENTATION: Update KMeans JSDoc from 'internal helper' to reflect public status
- [ ] #9 DOCUMENTATION: Add JSDoc @param/@returns/@example tags to KMeans SpectralClustering and AgglomerativeClustering classes
- [ ] #10 CODE QUALITY: Remove orphaned tf-adapter-global.browser.ts test file
- [ ] #11 CODE QUALITY: Remove dead code — findBMUOptimized (never called) and unused _centered/_rowCoords/_colCoords in SOM linear init
- [ ] #12 CODE QUALITY: Replace console.log calls with configurable logging or callbacks (som.ts lines 206 and 784)
- [ ] #13 CODE QUALITY: Replace dynamic require() calls with proper imports in kmeans.ts laplacian.ts and smallest_eigenvectors_with_values.ts
- [ ] #14 CODE QUALITY: Remove redundant .npmignore (files field in package.json takes precedence)
- [ ] #15 CODE QUALITY: Remove unused @types/puppeteer devDependency (puppeteer v24 ships own types)
- [ ] #16 CODE QUALITY: Extract shared input conversion utility for DataMatrix-to-Tensor2D used in every algorithm
- [ ] #17 CODE QUALITY: Extract shared validation input conversion (tensor+labels) used identically in all 3 validation files
- [ ] #18 CODE QUALITY: Fix KNN affinity docstring that says max(A At) but implementation does 0.5*(A+At)
- [ ] #19 CORRECTNESS: Fix cosineDistance to use per-row norms instead of global Frobenius norm (tensor.ts lines 66-76)
- [ ] #20 CORRECTNESS: Fix SOM dispose() to null out weights_ bmus_ and labels_ after disposal
- [ ] #21 CORRECTNESS: Fix SOM enableStreamingMode to not mutate readonly params
- [ ] #22 CORRECTNESS: Fix SOM getStreamingStats virtualEpoch calculation to match partialFit
- [ ] #23 CORRECTNESS: Fix makeBlobs randomState to actually pass seed to each tf.random call (currently no-op)
- [ ] #24 CORRECTNESS: Add nClusters > nSamples validation to SpectralClustering.fit and AgglomerativeClustering.fit
- [ ] #25 CORRECTNESS: Add empty tensor input validation (not just empty array) to AgglomerativeClustering
- [ ] #26 CORRECTNESS: Fix agglomerative single-sample path to not dispose caller tensor
- [ ] #27 CORRECTNESS: Clamp negative squared distances in KMeans (matching pairwiseEuclideanMatrix)
- [ ] #28 CORRECTNESS: Fix processStream with autoTrain=false to throw instead of silently logging
- [ ] #29 BUILD: Update ESLint to flat config format and typescript-eslint v8
- [ ] #30 BUILD: Add CI npm cache configuration
- [ ] #31 BUILD: Add Node.js 22 to CI matrix
- [ ] #32 BUILD: Fix benchmark workflow running twice on PR merge
- [ ] #33 PERFORMANCE: Use Set instead of Array.includes in KMeans++ seeding for O(1) lookups
- [ ] #34 PERFORMANCE: Move SOM topographicError to batch computation instead of per-sample
- [ ] #35 PERFORMANCE: Use dataSync instead of arraySync in silhouetteScore for memory efficiency
- [ ] #36 SECURITY: Run npm audit fix for devDependency vulnerabilities
- [ ] #37 SECURITY: Pin CDN TensorFlow.js version in browser benchmark with SRI hash
- [ ] #38 SECURITY: Add input dimension upper-bound validation to prevent OOM on spectral clustering
<!-- AC:END -->
