---
id: TASK-46
title: Add SOM.cluster() method and document SOM usage patterns
status: Done
assignee: []
created_date: '2026-03-20'
updated_date: '2026-03-20 22:09'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Real-world consumers (e.g., code-charter) need 2-12 meaningful clusters from SOM, but predict() returns raw BMU neuron indices (up to gridWidth*gridHeight distinct labels). Users must currently implement their own 2-phase approach (SOM + agglomerative on weight vectors). Additionally, partialFit() requires the same feature dimensionality as fit() but this is undocumented and unvalidated. getWeights() returns a tf.Tensor3D with no documented dispose contract — callers don't know if they own it or if som.dispose() invalidates it. These issues were identified by a real consumer integrating clustering-tfjs into code-charter.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 SOM class has a public cluster(nClusters) method that performs 2-phase grouping (agglomerative on weight vectors) and returns meaningful cluster labels
- [x] #2 SOM.cluster() is documented with JSDoc including usage example
- [x] #3 partialFit() validates that input feature dimensions match fit() dimensions and throws clear error on mismatch
- [x] #4 partialFit() JSDoc documents the same-dimensionality constraint
- [x] #5 getWeights() either returns number[][][] (no dispose needed) or documents tensor ownership contract clearly
- [x] #6 SOM.dispose() behavior documented w.r.t. previously returned tensors
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add SOMClusterOptions interface to types.ts\n2. Change getWeights() to return number[][][] with JSDoc\n3. Update dispose() with nulling + JSDoc\n4. Add cluster() method with agglomerative 2-phase clustering + JSDoc\n5. Add partialFit() dimension validation + JSDoc updates\n6. Update som_visualization.ts callers (5 functions)\n7. Update test files (som.test.ts, som.reference.test.ts, som_hexagonal.test.ts)\n8. Update docs (API.md, basic-usage.md) and examples (som-example.js)\n9. Run tests and linter
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
## Implementation Notes

### Approach
Implemented all 6 acceptance criteria in a single pass with comprehensive test coverage (26 new tests, 55 total in som.test.ts).

### Features implemented
- **cluster(nClusters, options?)**: 2-phase method that flattens SOM weight vectors into a 2D array, runs AgglomerativeClustering, then maps neuron cluster labels back to data points via BMU indices. Supports configurable linkage and metric via SOMClusterOptions.
- **partialFit() dimension validation**: Added else branch after weight initialization to compare xTensor.shape[1] against weights_.shape[2]. Error message includes both expected and actual dimensions.
- **getWeights() → number[][][]**: Changed return from Tensor3D reference to plain array snapshot via arraySync(). Eliminates tensor ownership ambiguity entirely.
- **dispose() improvements**: Now nulls out references after disposal (idempotent). JSDoc documents that getWeights() arrays survive disposal, getUMatrix() tensors are caller-owned.

### Modified files
- src/clustering/som.ts (cluster method, getWeights change, partialFit validation, dispose improvements)
- src/clustering/types.ts (SOMClusterOptions interface)
- src/utils/som_visualization.ts (5 callers updated for number[][][] return)
- test/clustering/som.test.ts (26 new tests)
- test/clustering/som.reference.test.ts (removed weights.dispose())
- test/clustering/som_hexagonal.test.ts (updated for plain array API)
- docs/API.md, docs/examples/basic-usage.md, examples/som-example.js, examples/node-basic.js

### Technical decisions
- Chose number[][][] over Tensor3D for getWeights() because every caller was immediately calling arraySync()/array() anyway, and it eliminates the dangerous shared-reference pattern where dispose() invalidates external references.
- Added SOMClusterOptions as a separate interface rather than reusing AgglomerativeClusteringParams to keep the SOM API self-contained.
<!-- SECTION:NOTES:END -->
