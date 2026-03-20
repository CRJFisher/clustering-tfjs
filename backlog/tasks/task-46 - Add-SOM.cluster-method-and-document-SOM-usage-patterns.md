---
id: TASK-46
title: Add SOM.cluster() method and document SOM usage patterns
status: In Progress
assignee: []
created_date: '2026-03-20'
updated_date: '2026-03-20 21:27'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Real-world consumers (e.g., code-charter) need 2-12 meaningful clusters from SOM, but predict() returns raw BMU neuron indices (up to gridWidth*gridHeight distinct labels). Users must currently implement their own 2-phase approach (SOM + agglomerative on weight vectors). Additionally, partialFit() requires the same feature dimensionality as fit() but this is undocumented and unvalidated. getWeights() returns a tf.Tensor3D with no documented dispose contract — callers don't know if they own it or if som.dispose() invalidates it. These issues were identified by a real consumer integrating clustering-tfjs into code-charter.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 SOM class has a public cluster(nClusters) method that performs 2-phase grouping (agglomerative on weight vectors) and returns meaningful cluster labels
- [ ] #2 SOM.cluster() is documented with JSDoc including usage example
- [ ] #3 partialFit() validates that input feature dimensions match fit() dimensions and throws clear error on mismatch
- [ ] #4 partialFit() JSDoc documents the same-dimensionality constraint
- [ ] #5 getWeights() either returns number[][][] (no dispose needed) or documents tensor ownership contract clearly
- [ ] #6 SOM.dispose() behavior documented w.r.t. previously returned tensors
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add SOMClusterOptions interface to types.ts\n2. Change getWeights() to return number[][][] with JSDoc\n3. Update dispose() with nulling + JSDoc\n4. Add cluster() method with agglomerative 2-phase clustering + JSDoc\n5. Add partialFit() dimension validation + JSDoc updates\n6. Update som_visualization.ts callers (5 functions)\n7. Update test files (som.test.ts, som.reference.test.ts, som_hexagonal.test.ts)\n8. Update docs (API.md, basic-usage.md) and examples (som-example.js)\n9. Run tests and linter
<!-- SECTION:PLAN:END -->
