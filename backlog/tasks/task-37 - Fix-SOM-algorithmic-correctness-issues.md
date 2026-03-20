---
id: task-37
title: Fix SOM algorithmic correctness issues
status: To Do
assignee: []
created_date: '2026-03-20'
labels: []
dependencies: []
---

## Description

The SOM implementation has multiple correctness bugs. (1) Weight update missing normalization by sum of influences — batch updates scale linearly with batch size instead of being normalized. (2) areNeighbors uses 4-connectivity for rectangular topology but getUMatrix uses 8-connectivity, causing inflated topographic error. (3) findSecondBMU uses Math.min(...spread) which causes stack overflow on grids larger than ~100x100. (4) No data shuffling between training epochs, causing order-dependent bias. (5) Linear initialization doesn't actually use PCA — computes centered data but never uses it, collapses to 1D. (6) getDensityMap Gaussian convolution is not implemented — returns raw hitMap despite computing a kernel.

## Acceptance Criteria

- [ ] Weight update normalizes by sum of influences for batch and mini-batch modes
- [ ] Rectangular topology uses consistent connectivity (4 or 8) for both areNeighbors and getUMatrix
- [ ] findSecondBMU uses iterative min-finding instead of Math.min spread
- [ ] Training epochs shuffle data or randomly sample
- [ ] Linear initialization uses actual PCA components to span 2D surface
- [ ] getDensityMap applies Gaussian convolution or is removed/marked unimplemented
- [ ] All fixes verified against MiniSom reference
