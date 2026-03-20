---
id: task-33
title: >-
  Fix memory leaks, broken SOM clustering path, and missing agglomerative
  features
status: To Do
assignee: []
created_date: '2026-03-20'
labels: []
dependencies: []
---

## Description

Five parallel investigation agents identified four confirmed bugs/gaps in the clustering-tfjs library that affect consumers like code-charter. The issues are: (1) findOptimalClusters leaks tensors because clusterer instances are never disposed after fitPredict — spectral clustering leaks 2 tensors per k iteration (affinity matrix + KMeans centroids), KMeans leaks centroids, SOM leaks weights/grid/BMUs. (2) findOptimalClusters with algorithm 'som' is broken by design — the k parameter loosely controls grid size but the output always has gridW*gridH labels, making the optimal-k sweep meaningless. A secondary grouping step (clustering SOM weight vectors) is needed but entirely absent. (3) AgglomerativeClustering computes merge distances (minDist) in the inner loop but discards them — only children_ is stored, not distances_, blocking dendrogram cutting and distance-threshold stopping. (4) AgglomerativeClustering lacks metric 'precomputed' support for consuming pre-built distance/similarity matrices, unlike SpectralClustering which supports affinity 'precomputed'. The validation metrics (silhouette, Davies-Bouldin, Calinski-Harabasz) were confirmed to have NO tensor leaks — they properly use tf.tidy().

## Acceptance Criteria

- [ ] Add dispose() method to KMeans that cleans up centroids_ tensor
- [ ] Call clusterer.dispose() after extracting labels in findOptimalClusters loop (after line 187) for all algorithms that have dispose()
- [ ] Fix findOptimalClusters SOM path to use two-phase clustering: train SOM then apply agglomerative/kmeans on weight vectors to produce exactly k clusters
- [ ] Add distances_ property to AgglomerativeClustering that records merge distance at each step (capture minDist before it is discarded after line 152)
- [ ] Add distanceThreshold parameter to AgglomerativeClusteringParams as alternative stopping criterion to nClusters
- [ ] Add metric precomputed to AgglomerativeClustering that bypasses pairwiseDistanceMatrix and uses input directly as distance matrix
- [ ] Validate precomputed distance matrix is square symmetric with zero diagonal
- [ ] Disallow linkage ward with metric precomputed (matching scikit-learn)
- [ ] Add tensor leak regression tests using tf.memory().numTensors assertions in findOptimalClusters tests
- [ ] Add tests for SOM two-phase clustering producing correct cluster count
- [ ] Add tests for agglomerative distances_ output matching expected merge distances
- [ ] Add tests for agglomerative with metric precomputed producing same results as computing distances internally

## Implementation Plan

### Bug 1: Memory Leaks in findOptimalClusters (HIGH PRIORITY)

**Root cause:** `src/utils/findOptimalClusters.ts` lines 114-198 create clusterer instances in a loop but never call `dispose()` after `fitPredict` returns. No `tf.tidy()` wraps the loop.

**Leaked tensors per k iteration:**
- SpectralClustering: `affinityMatrix_` (n×n tensor) + internal KMeans `centroids_` = 2 tensors
- KMeans: `centroids_` (k×d tensor) = 1 tensor (KMeans has NO dispose() method)
- SOM: `weights_` + `gridDistanceMatrix_` + `bmus_` = 3 tensors
- AgglomerativeClustering: 0 tensors (uses plain arrays internally)

**Fix steps:**

1. **Add `dispose()` to KMeans** (`src/clustering/kmeans.ts`):
   ```typescript
   public dispose(): void {
     if (this.centroids_) {
       this.centroids_.dispose();
       this.centroids_ = null;
     }
   }
   ```

2. **Add clusterer disposal in findOptimalClusters** (`src/utils/findOptimalClusters.ts`), after extracting labels (~line 187):
   ```typescript
   // After: const labels = Array.from(labelsArray);
   if (typeof (clusterer as any).dispose === 'function') {
     (clusterer as any).dispose();
   }
   ```

3. **Fix SpectralClustering.dispose()** to also clean up the internal KMeans centroids (currently it only disposes `affinityMatrix_`).

4. **Add tensor leak regression tests** in `test/utils/findOptimalClusters.test.ts`:
   ```typescript
   it('should not leak tensors across k iterations', async () => {
     const before = tf.memory().numTensors;
     await findOptimalClusters(data, { algorithm: 'spectral', ... });
     const after = tf.memory().numTensors;
     expect(after).toBeLessThanOrEqual(before + 2); // small tolerance for input tensor
   });
   ```

### Bug 2: Broken SOM Path in findOptimalClusters (HIGH PRIORITY)

**Root cause:** `src/utils/findOptimalClusters.ts` lines 133-145 map `k` to a grid size via `gridSize = ceil(sqrt(k))`, but SOM always produces `gridW*gridH` labels (not `k`). For k=3: creates 2×2 grid → 4 labels. For k=5: creates 3×2 grid → 6 labels. The validation metrics then score neuron assignments, not meaningful clusters.

**Fix: Add two-phase SOM clustering.**

After SOM training, apply secondary clustering on the SOM weight vectors to produce exactly `k` clusters:

1. In the `'som'` case of findOptimalClusters, after `fitPredict`:
   ```typescript
   case 'som': {
     // Phase 1: Train SOM with fixed grid (e.g., 5×5)
     const grid_size = Math.max(3, Math.ceil(Math.sqrt(n_samples)));
     const som = new SOM({
       nClusters: grid_size * grid_size,
       gridWidth: grid_size,
       gridHeight: grid_size,
       ...algorithmParams,
     });
     await som.fit(dataTensor);

     // Phase 2: Cluster the weight vectors to get k macro-clusters
     const weights = som.getWeights(); // [gridH, gridW, features]
     const weight_matrix = reshape_to_2d(weights); // [gridH*gridW, features]
     const agg = new AgglomerativeClustering({ nClusters: k, linkage: 'ward' });
     const neuron_labels = await agg.fitPredict(weight_matrix);

     // Phase 3: Re-map data point labels through neuron → macro-cluster
     const bmu_labels = som.labels_; // neuron indices per data point
     const final_labels = bmu_labels.map(neuron_idx => neuron_labels[neuron_idx]);

     som.dispose();
     break;
   }
   ```

2. Add a utility function `som_to_clusters(som, k)` that encapsulates this two-phase logic for reuse outside `findOptimalClusters`.

### Bug 3: Missing Merge Distances in AgglomerativeClustering (MEDIUM)

**Root cause:** `src/clustering/agglomerative.ts` line 152 stores `children.push([idI, idJ])` but `minDist` (computed at lines 134-147) is discarded.

**Fix (~6 lines of production code):**

1. Add property declaration (after line 43):
   ```typescript
   public distances_: number[] | null = null;
   ```

2. Initialize array (after line 121):
   ```typescript
   const merge_distances: number[] = [];
   ```

3. Capture distance (after line 152):
   ```typescript
   merge_distances.push(minDist);
   ```

4. Store result (after line 186):
   ```typescript
   this.distances_ = merge_distances;
   ```

5. Handle trivial case (line 95-99):
   ```typescript
   this.distances_ = [];
   ```

6. **Add `distanceThreshold` parameter** to `AgglomerativeClusteringParams` in `src/clustering/types.ts`:
   ```typescript
   distanceThreshold?: number; // mutually exclusive with nClusters
   ```

7. Change loop condition (line 130):
   ```typescript
   while (clusterIds.length > (distanceThreshold != null ? 1 : nClusters)) {
     // ... find minDist ...
     if (distanceThreshold != null && minDist > distanceThreshold) break;
     // ... rest of merge logic ...
   }
   ```

8. Update validation to allow either `nClusters` or `distanceThreshold` (not both).

### Bug 4: Missing Precomputed Matrix Support in AgglomerativeClustering (MEDIUM)

**Root cause:** `metric` only accepts `'euclidean' | 'manhattan' | 'cosine'`. The `fit()` method always calls `pairwiseDistanceMatrix(points, metric)` to compute distances.

**Fix:**

1. Add `'precomputed'` to types (`src/clustering/types.ts` line 161):
   ```typescript
   metric?: 'euclidean' | 'manhattan' | 'cosine' | 'precomputed';
   ```

2. Add `'precomputed'` to `VALID_METRICS` (`src/clustering/agglomerative.ts` lines 58-62).

3. Add validation in `validateParams`:
   - Disallow `linkage: 'ward'` with `metric: 'precomputed'`

4. Add conditional in `fit()` (before line 108):
   ```typescript
   if (this.params.metric === 'precomputed') {
     // Validate: square, symmetric, zero diagonal
     const D_raw = isTensor(X) ? await X.array() : X;
     // ... validation ...
     D = D_raw as number[][];
     nSamples = D.length;
   } else {
     const distanceTensor = pairwiseDistanceMatrix(points, metric);
     D = await distanceTensor.array();
     distanceTensor.dispose();
   }
   ```

5. **Linkage formulas need NO changes** — they operate on the distance matrix regardless of how it was computed.

6. **Note:** `findOptimalClusters` validation metrics will produce nonsensical scores for precomputed matrices (they'd compute "distances between rows of the distance matrix"). This is a pre-existing design issue that also affects spectral with `affinity: 'precomputed'` and should be addressed separately.

### Implementation Order

1. Bug 1 (memory leaks) — highest impact, affects every consumer
2. Bug 2 (SOM path) — blocking for SOM adoption in code-charter
3. Bug 3 (merge distances) — enables dendrogram features in code-charter
4. Bug 4 (precomputed metric) — enables code-charter's combined matrix with agglomerative

### Key Files

| File | Changes |
|------|---------|
| `src/utils/findOptimalClusters.ts` | Add clusterer disposal, fix SOM two-phase path |
| `src/clustering/kmeans.ts` | Add `dispose()` method |
| `src/clustering/spectral.ts` | Fix `dispose()` to also clean internal KMeans centroids |
| `src/clustering/agglomerative.ts` | Add `distances_`, `distanceThreshold`, `metric: 'precomputed'` |
| `src/clustering/types.ts` | Add `distanceThreshold`, expand `metric` union type |
| `test/utils/findOptimalClusters.test.ts` | Tensor leak regression tests, SOM two-phase tests |
| `test/clustering/agglomerative.test.ts` | Tests for `distances_`, `distanceThreshold`, precomputed |
