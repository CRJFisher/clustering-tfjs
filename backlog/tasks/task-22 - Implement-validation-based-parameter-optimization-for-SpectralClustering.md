---
id: task-22
title: Implement validation-based parameter optimization for SpectralClustering
status: To Do
assignee: []
created_date: '2025-07-21'
labels: []
dependencies:
  - task-13
  - task-14
  - task-15
---

## Description

Use internal validation metrics (Calinski-Harabasz, Silhouette, Davies-Bouldin) to automatically optimize parameters and select best clustering results, especially for 3+ cluster cases where k-means initialization is sensitive

## Acceptance Criteria

- [ ] Implement parameter grid search using validation metrics
- [ ] Add useValidation flag to SpectralClustering params
- [ ] For 3+ clusters run multiple k-means attempts and select best by validation score
- [ ] Achieve ≥0.95 ARI on all 3-cluster test fixtures
- [ ] Document the optimization approach

## Implementation Plan

### Primary Metric: Calinski-Harabasz Index

Based on analysis of the three validation metrics, Calinski-Harabasz (CH) Index is the optimal choice for parameter optimization due to:

1. **Computational Efficiency**: O(n·k) complexity makes it ~100x faster than Silhouette
2. **Clear Optimization Direction**: Higher values consistently indicate better clustering
3. **Suitable for Small k**: For 3-cluster problems, CH index's bias towards more clusters is less problematic
4. **Practical for Parameter Tuning**: Fast evaluation enables testing many configurations

### Optimization Strategy

```typescript
// Pseudo-code for optimization approach
async function optimizeSpectralClustering(
  X: DataMatrix,
  baseParams: SpectralClusteringParams,
) {
  let bestScore = -Infinity;
  let bestLabels = null;

  // For RBF affinity, try different gamma values
  const gammaRange = baseParams.gamma
    ? [baseParams.gamma * 0.5, baseParams.gamma, baseParams.gamma * 2]
    : [0.1, 0.5, 1.0, 5.0, 10.0];

  for (const gamma of gammaRange) {
    // Compute embedding with this gamma
    const embedding = await computeSpectralEmbedding(X, {
      ...baseParams,
      gamma,
    });

    // Try multiple k-means runs (different seeds)
    for (let seed = 0; seed < 20; seed++) {
      const labels = await runKMeans(embedding, {
        nClusters: baseParams.nClusters,
        randomState: baseParams.randomState + seed,
        nInit: 1, // Single run per seed
      });

      // Fast evaluation with CH index
      const score = calinskiHarabasz(embedding, labels);

      if (score > bestScore) {
        bestScore = score;
        bestLabels = labels;
      }
    }
  }

  return bestLabels;
}
```

### Implementation Details

1. **Parameter Grid Search**
   - Gamma values for RBF: Test multiple scales around the provided/default value
   - Number of eigenvectors: Could try nClusters ± 1 to see if including more helps
   - K-means seeds: Try 20-30 different random seeds for 3+ cluster problems

2. **Validation Hierarchy**
   - **Primary**: Calinski-Harabasz for optimization loop (fast)
   - **Secondary**: Silhouette score for final validation (accurate but slow)
   - **Tertiary**: Davies-Bouldin as a sanity check

3. **Special Handling for 3+ Clusters**

   ```typescript
   if (this.params.nClusters >= 3 && this.params.useValidation) {
     // Multiple attempts with validation scoring
     let bestLabels = null;
     let bestScore = -Infinity;

     for (let attempt = 0; attempt < 20; attempt++) {
       const km = new KMeans({
         nClusters: this.params.nClusters,
         randomState: this.params.randomState + attempt,
         nInit: 1,
       });

       const labels = await km.fit(embedding);
       const score = calinskiHarabasz(embedding, labels);

       if (score > bestScore) {
         bestScore = score;
         bestLabels = labels;
       }
     }

     this.labels_ = bestLabels;
   }
   ```

### Expected Impact

For the failing 3-cluster tests:

- **circles_n3_knn**: Currently 0.899 → Target ≥0.95
- **circles_n3_rbf**: Currently 0.722 → Target ≥0.95
- **moons_n3_rbf**: Currently 0.946 → Target ≥0.95

By selecting the best clustering from multiple attempts based on CH score rather than relying on random initialization luck, we should achieve the target ARI threshold.

### Why This Will Work

Our investigation in task 12.25 showed that:

- Different random seeds produce wildly different results (ARI 0.24 to 0.95)
- sklearn achieves perfect results with lucky parameter combinations
- The embeddings are correct; it's purely a k-means initialization issue

By trying many initializations and selecting based on validation score, we'll find the same "lucky" configurations that sklearn sometimes finds, but reliably rather than randomly.
