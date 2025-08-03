---
id: task-27
title: Fix Windows Jest configuration for TensorFlow backend
status: Done
assignee:
  - '@assistant'
created_date: '2025-08-03'
updated_date: '2025-08-03'
labels: []
dependencies: []
---

## Description

The Jest moduleNameMapper redirects tfjs-node to tfjs on Windows CI, but tests fail with 'Backend name tensorflow not found in registry' error

## Acceptance Criteria

- [ ] Windows tests pass without backend errors
- [ ] Tests can use appropriate TensorFlow backends on Windows
- [ ] No regression in test functionality

## Implementation Notes

Fixed Windows CI TensorFlow backend issues by:
1. Updated tf-adapter.ts to dynamically load tfjs or tfjs-node based on platform
2. Modified test/setup.js to set TF_FORCE_CPU_BACKEND env var on Windows CI
3. Updated test/tensorflow-helper.ts to use the env var for backend selection
4. Fixed findOptimalClusters.test.ts to detect available backends dynamically
5. Removed Jest moduleNameMapper approach as it wasn't working properly
