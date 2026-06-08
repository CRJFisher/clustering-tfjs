---
id: task-37
title: Fix SOM algorithmic correctness issues
status: Done
assignee: [claude]
created_date: '2026-03-20'
labels: []
dependencies: []
---

## Description

The SOM implementation has multiple correctness bugs. (1) Weight update missing normalization by sum of influences — batch updates scale linearly with batch size instead of being normalized. (2) areNeighbors uses 4-connectivity for rectangular topology but getUMatrix uses 8-connectivity, causing inflated topographic error. (3) findSecondBMU uses Math.min(...spread) which causes stack overflow on grids larger than ~100x100. (4) No data shuffling between training epochs, causing order-dependent bias. (5) Linear initialization doesn't actually use PCA — computes centered data but never uses it, collapses to 1D. (6) getDensityMap Gaussian convolution is not implemented — returns raw hitMap despite computing a kernel.

## Acceptance Criteria

- [x] Weight update normalizes by sum of influences for batch and mini-batch modes
- [x] Rectangular topology uses consistent connectivity (4 or 8) for both areNeighbors and getUMatrix
- [x] Second-BMU search avoids `Math.min(...spread)` (no stack overflow on large grids)
- [x] Training epochs shuffle data or randomly sample
- [x] Linear initialization uses actual PCA components to span 2D surface
- [x] getDensityMap applies Gaussian convolution or is removed/marked unimplemented
- [x] All fixes verified against MiniSom reference

## Implementation Plan

1. Fix weight update normalization in `updateWeights` — divide totalUpdate by sum of influences per neuron
2. Standardize rectangular topology on 8-connectivity to match MiniSom — fix `areNeighbors` and `getNeighborDistanceMatrix`
3. Replace `Math.min(...spread)` in `findSecondBMU` with iterative loop
4. Add Fisher-Yates shuffling with seeded RNG in `fitTensor` training loop
5. Replace broken linear init (1D collapse) with PCA-based 2D surface initialization
6. Complete `getDensityMap` Gaussian convolution using `tf.conv2d`
7. Add tests for all 6 fixes, run full test suite

## Implementation Notes

### Approach

All 6 correctness bugs were fixed in the core SOM files, with 7 new tests added to verify each fix.

### Fixes implemented

1. **Weight normalization** (`som.ts:updateWeights`): Added `influence.sum(0)` normalization with epsilon=1e-8 guard. Learning rate now applied after normalization rather than before, matching the formula `Δw = lr * Σ(h*(x-w)) / Σ(h)`.

2. **8-connectivity** (`som.ts:areNeighbors`, `som_visualization.ts:getNeighborDistanceMatrix`): Standardized on 8-connectivity for rectangular topology to match MiniSom reference, `getUMatrix`, and `getNeighbors`. The `areNeighbors` check now uses `rowDiff <= 1 && colDiff <= 1 && (rowDiff + colDiff > 0)`.

3. **Second-BMU search** (`som.ts:topographic_error`): The second-best matching unit is computed fully vectorized — `argMin` over the distance grid, one-hot masking of the first BMU, then `argMin` again — eliminating the `Math.min(...spread)` that overflowed the call stack on grids larger than ~100×100. (A standalone `findSecondBMU` helper is no longer needed.)

4. **Epoch shuffling** (`som.ts:fitTensor`): Added Fisher-Yates shuffle via `make_random_stream` from the project's seeded RNG. Each epoch creates shuffled indices, uses `tf.gather` to reorder data, then disposes. Deterministic when `randomState` is set.

5. **Linear PCA init** (`som_neighborhood.ts:initialize_weights 'linear'`): Replaced broken 1D interpolation (`alpha*0.7 + beta*0.3`) with proper PCA-based initialization: computes covariance, extracts top-2 eigenvectors via power iteration, scales by projection std, and spans a 2D surface from -1 to +1 along each PC (matching MiniSom's `pca_weights_init`).

6. **getDensityMap convolution** (`som_visualization.ts`): Completed the Gaussian convolution using `tf.conv2d` with 'same' padding. The hitMap is properly reshaped to [1,H,W,1], convolved with the kernel [K,K,1,1], and squeezed back. Added proper `hitMap.dispose()` in finally block.

### Additional changes

- Added `getDensityMap` and `getNeighborDistanceMatrix` to public exports in `src/index.ts`
- Added import for `make_random_stream` and `RandomStream` in `som.ts`
- Added `shuffleIndices` private method to `SOM` class

### Modified files

- `src/clustering/som.ts` — weight normalization, 8-connectivity, epoch shuffling, shuffle helper
- `src/clustering/som_neighborhood.ts` — PCA-based linear initialization, 8-connectivity neighbor helpers
- `src/utils/som_visualization.ts` — getDensityMap convolution, 8-connectivity in getNeighborDistanceMatrix
- `src/index.ts` — added exports
- `test/clustering/som.test.ts` — 7 new tests for all fixes
