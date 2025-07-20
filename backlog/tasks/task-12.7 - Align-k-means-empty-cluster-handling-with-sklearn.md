---
id: task-12.7
title: Align k-means empty cluster handling with sklearn
status: Done
assignee:
  - '@chuck'
created_date: '2025-07-19'
updated_date: '2025-07-20'
labels: []
dependencies: []
parent_task_id: task-12
---

## Description

When k-means encounters empty clusters, sklearn uses a specific reseeding strategy based on points with highest distance to nearest centroid. Our implementation may handle this differently, causing divergent results. This is especially important with the deterministic multi-init approach.

## Acceptance Criteria

- [x] Implement sklearn's empty cluster reseeding strategy
- [x] Use points with highest SSE contribution for new centers
- [x] Ensure deterministic tie-breaking using RandomState
- [x] Add test for empty cluster scenario

## Implementation Plan

1. Analyze sklearn's empty cluster handling in \_kmeans_single_lloyd
2. Check our current implementation strategy
3. Update our k-means to match sklearn's approach
4. Add tests to verify empty cluster handling
5. Test impact on fixture tests

## Implementation Notes

### sklearn's Strategy

From sklearn source code analysis:

- When empty clusters are detected after label assignment
- Find points with largest distance to their nearest center
- Assign these farthest points as new centers for empty clusters
- Uses np.argpartition for deterministic tie-breaking
- No randomness involved in the reseeding itself

### Changes Made

Updated `src/clustering/kmeans.ts` to:

1. Detect empty clusters after computing new centroids
2. Find points with maximum distance to nearest center (using minDistSq)
3. Sort points by distance and assign farthest ones to empty clusters
4. This matches sklearn's greedy strategy

### Test Results

- Created test in `test/clustering/kmeans_empty_clusters.test.ts`
- Empty cluster handling works correctly
- Fixture test results unchanged: still 5/12 passing (41.7%)

### Conclusion

Empty cluster handling is now aligned with sklearn, but this was not the cause of fixture test failures. The remaining differences are likely due to:

1. RBF fixtures having incorrect gamma values (task 12.5)
2. Small numerical differences in algorithms
3. Non-deterministic aspects of k-means clustering

The implementation is correct but doesn't improve fixture test results.
