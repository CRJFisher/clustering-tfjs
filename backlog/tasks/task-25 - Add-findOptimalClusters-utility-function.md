---
id: task-25
title: Add findOptimalClusters utility function
status: Done
assignee: []
created_date: '2025-07-30'
updated_date: '2025-07-30'
labels: []
dependencies: []
---

## Description

Implement a utility function that automatically determines the optimal number of clusters for a dataset by evaluating multiple k values using validation metrics

## Acceptance Criteria

- [x] findOptimalClusters function implemented in src/utils/
- [x] Function supports all clustering algorithms (KMeans/Spectral/Agglomerative)
- [x] Configurable k range and validation metrics to use
- [x] Returns detailed results including all metric scores
- [x] Export function from main index for easy access
- [x] Unit tests with various datasets and edge cases
- [x] TypeScript types for function parameters and return values
- [x] Documentation in code with JSDoc comments
- [x] README updated to show built-in function usage

## Implementation Notes

Implemented a comprehensive `findOptimalClusters` utility function that automates the process of finding the optimal number of clusters for a dataset.

Key features implemented:
- Supports all three clustering algorithms (KMeans, Spectral, Agglomerative)
- Configurable min/max cluster range
- Option to select which validation metrics to use
- Custom scoring function support
- Proper tensor memory management
- Comprehensive test suite with edge cases

Fixed issues in validation functions:
- Corrected tensor disposal logic in all three validation metrics (silhouette, Davies-Bouldin, Calinski-Harabasz)
- Fixed backwards logic that was disposing user-provided tensors incorrectly

The function is now exported from the main index and documented in the README with both basic and advanced usage examples.
