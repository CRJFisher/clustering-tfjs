# Recommendations for Next Steps on Task 12

## Key Findings from Task 12.4.1

1. **Eigensolver accuracy is NOT the bottleneck** - The improved eigensolver fixed numerical issues but didn't improve test results
2. **Most failures are close** - 5 out of 7 failing tests have ARI > 0.7 (many > 0.84)
3. **Pattern in failures**:
   - Only 1 RBF test has very low ARI (blobs_n2_rbf: 0.064)
   - All circle datasets fail (both RBF and k-NN)
   - Moons RBF tests are close to passing (0.869, 0.841)

## Assessment of Current Subtasks

### Task 12.5: Remove zero-padding of embedding
**Relevance: LOW** - We already handle this correctly by throwing an error when informative vectors < nClusters

### Task 12.6: Align k-means empty cluster handling
**Relevance: MEDIUM** - Could be relevant but unlikely to fix the systematic failures we're seeing

### Task 12.7: Complete randomState propagation
**Relevance: LOW** - The Jacobi solver doesn't use randomness, and our results are very consistent

## Recommended New Subtasks

### 1. **Task 12.8: Debug RBF gamma scaling** (HIGH PRIORITY)
The blobs_n2_rbf test has ARI=0.064 which suggests a fundamental issue with RBF affinity calculation. Sklearn may handle gamma differently or have different default scaling.

**Investigation points:**
- Compare our RBF kernel computation with sklearn's
- Check if sklearn does any data preprocessing before computing RBF
- Verify gamma parameter interpretation

### 2. **Task 12.9: Analyze spectral embedding differences** (HIGH PRIORITY)
Since eigensolver accuracy isn't the issue, the problem may be in how we construct or use the embedding.

**Investigation points:**
- Dump spectral embeddings from both implementations and compare
- Check if sklearn applies any normalization we're missing
- Verify the exact eigenvectors being used (which columns selected)

### 3. **Task 12.10: Compare affinity matrices directly** (MEDIUM PRIORITY)
The circle dataset failures across both RBF and k-NN suggest the issue might be earlier in the pipeline.

**Investigation points:**
- Export affinity matrices from sklearn and compare element-wise
- Check for subtle differences in k-NN graph construction
- Verify symmetrization approach matches sklearn

### 4. **Task 12.11: Investigate k-means initialization details** (LOW PRIORITY)
Only if other approaches fail. The consistent results suggest this isn't the main issue.

## Recommendation

I recommend:
1. **Demote tasks 12.5 and 12.7** - They're unlikely to help
2. **Keep task 12.6** but lower priority
3. **Create tasks 12.8 and 12.9** as the primary investigation paths
4. **Focus on the blobs_n2_rbf test** (ARI=0.064) as it likely has the most obvious discrepancy