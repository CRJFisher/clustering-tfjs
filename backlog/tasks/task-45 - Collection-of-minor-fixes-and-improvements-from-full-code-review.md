---
id: task-45
title: Collection of minor fixes and improvements from full code review
status: To Do
assignee: []
created_date: '2026-03-20'
labels: []
dependencies: []
---

## Description

This task collects smaller improvements identified during the comprehensive 15-agent code review that don't warrant individual tasks. These should be addressed opportunistically or as a cleanup pass.

## Acceptance Criteria

- [ ] DOCUMENTATION: Update CHANGELOG.md for versions 0.2.0 through 0.4.0
- [ ] DOCUMENTATION: Remove predict() from README common interface (only SOM has it)
- [ ] DOCUMENTATION: Remove non-existent init parameter from KMeans docs in README and API.md
- [ ] DOCUMENTATION: Add SOM to README API Reference section
- [ ] DOCUMENTATION: Fix broken links to non-existent performance.md and migration-guide.md
- [ ] DOCUMENTATION: Remove stale 'coming soon' and 'backend selection coming in future' comments from README
- [ ] DOCUMENTATION: Update placeholder author in package.json
- [ ] DOCUMENTATION: Update KMeans JSDoc from 'internal helper' to reflect public status
- [ ] DOCUMENTATION: Add JSDoc @param/@returns/@example tags to KMeans SpectralClustering and AgglomerativeClustering classes
- [ ] CODE QUALITY: Remove orphaned tf-adapter-global.browser.ts test file
- [ ] CODE QUALITY: Remove dead code — findBMUOptimized (never called) and unused _centered/_rowCoords/_colCoords in SOM linear init
- [ ] CODE QUALITY: Replace console.log calls with configurable logging or callbacks (som.ts lines 206 and 784)
- [ ] CODE QUALITY: Replace dynamic require() calls with proper imports in kmeans.ts laplacian.ts and smallest_eigenvectors_with_values.ts
- [ ] CODE QUALITY: Remove redundant .npmignore (files field in package.json takes precedence)
- [ ] CODE QUALITY: Remove unused @types/puppeteer devDependency (puppeteer v24 ships own types)
- [ ] CODE QUALITY: Extract shared input conversion utility for DataMatrix-to-Tensor2D used in every algorithm
- [ ] CODE QUALITY: Extract shared validation input conversion (tensor+labels) used identically in all 3 validation files
- [ ] CODE QUALITY: Fix KNN affinity docstring that says max(A At) but implementation does 0.5*(A+At)
- [ ] CORRECTNESS: Fix cosineDistance to use per-row norms instead of global Frobenius norm (tensor.ts lines 66-76)
- [ ] CORRECTNESS: Fix SOM dispose() to null out weights_ bmus_ and labels_ after disposal
- [ ] CORRECTNESS: Fix SOM enableStreamingMode to not mutate readonly params
- [ ] CORRECTNESS: Fix SOM getStreamingStats virtualEpoch calculation to match partialFit
- [ ] CORRECTNESS: Fix makeBlobs randomState to actually pass seed to each tf.random call (currently no-op)
- [ ] CORRECTNESS: Add nClusters > nSamples validation to SpectralClustering.fit and AgglomerativeClustering.fit
- [ ] CORRECTNESS: Add empty tensor input validation (not just empty array) to AgglomerativeClustering
- [ ] CORRECTNESS: Fix agglomerative single-sample path to not dispose caller tensor
- [ ] CORRECTNESS: Clamp negative squared distances in KMeans (matching pairwiseEuclideanMatrix)
- [ ] CORRECTNESS: Fix processStream with autoTrain=false to throw instead of silently logging
- [ ] BUILD: Update ESLint to flat config format and typescript-eslint v8
- [ ] BUILD: Add CI npm cache configuration
- [ ] BUILD: Add Node.js 22 to CI matrix
- [ ] BUILD: Fix benchmark workflow running twice on PR merge
- [ ] PERFORMANCE: Use Set instead of Array.includes in KMeans++ seeding for O(1) lookups
- [ ] PERFORMANCE: Move SOM topographicError to batch computation instead of per-sample
- [ ] PERFORMANCE: Use dataSync instead of arraySync in silhouetteScore for memory efficiency
- [ ] SECURITY: Run npm audit fix for devDependency vulnerabilities
- [ ] SECURITY: Pin CDN TensorFlow.js version in browser benchmark with SRI hash
- [ ] SECURITY: Add input dimension upper-bound validation to prevent OOM on spectral clustering
