---
id: task-46
title: Add SOM.cluster() method and document SOM usage patterns
status: To Do
assignee: []
created_date: '2026-03-20'
labels: []
dependencies: []
---

## Description

Real-world consumers (e.g., code-charter) need 2-12 meaningful clusters from SOM, but predict() returns raw BMU neuron indices (up to gridWidth*gridHeight distinct labels). Users must currently implement their own 2-phase approach (SOM + agglomerative on weight vectors). Additionally, partialFit() requires the same feature dimensionality as fit() but this is undocumented and unvalidated. getWeights() returns a tf.Tensor3D with no documented dispose contract — callers don't know if they own it or if som.dispose() invalidates it. These issues were identified by a real consumer integrating clustering-tfjs into code-charter.

## Acceptance Criteria

- [ ] SOM class has a public cluster(nClusters) method that performs 2-phase grouping (agglomerative on weight vectors) and returns meaningful cluster labels
- [ ] SOM.cluster() is documented with JSDoc including usage example
- [ ] partialFit() validates that input feature dimensions match fit() dimensions and throws clear error on mismatch
- [ ] partialFit() JSDoc documents the same-dimensionality constraint
- [ ] getWeights() either returns number[][][] (no dispose needed) or documents tensor ownership contract clearly
- [ ] SOM.dispose() behavior documented w.r.t. previously returned tensors
