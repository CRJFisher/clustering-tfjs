---
id: task-24.1
title: 'Phase 1: Decouple core logic from tfjs-node'
status: Done
assignee:
  - '@chuck'
created_date: '2025-07-30'
updated_date: '2025-08-02'
labels: []
dependencies: []
parent_task_id: task-24
---

## Description

Remove all hardcoded tfjs-node imports and create the platform adapter pattern foundation

## Acceptance Criteria

- [x] All direct imports of @tensorflow/tfjs-node removed
- [x] tf-adapter.ts module created
- [x] Core algorithms refactored to use tf-adapter
- [x] @tensorflow/tfjs-core added to dependencies
- [x] All tfjs backends moved to peer dependencies

## Implementation Plan

1. Search for all @tensorflow/tfjs-node imports in src/
2. Create src/tf-adapter.ts with a platform-agnostic TensorFlow interface
3. Update all algorithms to import from tf-adapter instead of tfjs-node directly
4. Update package.json dependencies
5. Run tests to ensure nothing breaks

## Implementation Notes

Created tf-adapter.ts module that re-exports all TensorFlow functionality. Updated all 22 source files to import from tf-adapter instead of directly from @tensorflow/tfjs-node. Updated package.json to use @tensorflow/tfjs-core as a dependency and moved all TensorFlow backends to peer dependencies. All tests pass and build succeeds.
