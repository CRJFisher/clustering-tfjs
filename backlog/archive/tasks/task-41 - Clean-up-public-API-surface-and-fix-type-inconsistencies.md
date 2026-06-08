---
id: task-41
title: Clean up public API surface and fix type inconsistencies
status: To Do
assignee: []
created_date: '2026-03-20'
labels: []
dependencies: []
---

## Description

The public API leaks internal implementation details and has inconsistent patterns. Internal functions like jacobi_eigen_decomposition, smallest_eigenvectors, deterministic_eigenpair_processing, findBMU, findBMUBatch, and individual neighborhood functions are exported. Snake_case and camelCase are mixed in exports. SOMParams inherits meaningless required nClusters from BaseClusteringParams. SpectralClustering captureDebugInfo is hidden via type intersection not in exported params. fitPredict returns LabelVector union but always returns number[]. The Clustering namespace object is not typed against ClusteringNamespace interface. src/clustering.ts and src/index.ts have overlapping re-exports.

## Acceptance Criteria

- [ ] Internal implementation functions removed from public exports
- [ ] All public API functions use consistent camelCase naming
- [ ] SOMParams does not require nClusters (uses Omit or separate base type)
- [ ] captureDebugInfo added to SpectralClusteringParams interface
- [ ] fitPredict return type narrowed to Promise<number[]>
- [ ] Single canonical export barrel in src/index.ts without overlap from src/clustering.ts
- [ ] Subpath exports configured in package.json for utils and validation
