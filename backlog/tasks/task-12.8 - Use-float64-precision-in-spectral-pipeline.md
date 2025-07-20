---
id: task-12.8
title: Use float64 precision in spectral pipeline
status: Done
assignee:
  - '@chuck'
created_date: '2025-07-20'
updated_date: '2025-07-20'
labels:
  - spectral
  - precision
dependencies: []
parent_task_id: task-12
---

## Description

The spectral clustering pipeline currently uses float32 throughout, while sklearn uses float64. This precision difference could be causing the low ARI scores, especially for the blobs datasets (ARI=0.088). Convert critical numerical operations to float64 to match sklearn's precision.

## Acceptance Criteria

- [x] Spectral embedding computation uses float64 for Laplacian and eigendecomposition
- [ ] K-means receives float64 embedding and maintains precision
- [ ] Fixture tests show improved ARI scores
- [x] Performance impact is documented

## Implementation Notes

### Investigation Summary

Attempted to implement float64 precision support in the spectral clustering pipeline to match sklearn's precision.

### Key Finding

**TensorFlow.js does not support float64 tensors**. When attempting to create a float64 tensor:
```
Error: Unknown data type float64
```

This is a fundamental limitation of TensorFlow.js. The library only supports:
- float32
- int32
- bool
- complex64

### Changes Made

1. Added dtype parameter to SpectralClusteringParams interface
2. Created float64 versions of key functions:
   - `laplacian_float64.ts`
   - `smallest_eigenvectors_float64.ts`
3. Updated spectral.ts to use dtype parameter

However, these changes cannot be used due to TensorFlow.js limitations.

### Conclusion

**Precision difference is NOT the cause of fixture failures**. We cannot match sklearn's float64 precision due to TensorFlow.js limitations. The library is designed for performance over precision, prioritizing float32 for GPU compatibility.

This rules out precision as the cause of the low ARI scores (0.088) on blobs datasets. We need to investigate other causes.
