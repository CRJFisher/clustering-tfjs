---
id: TASK-43
title: Fix TensorFlow.js backend architecture and platform detection
status: Done
assignee:
  - '@claude'
created_date: '2026-03-20'
updated_date: '2026-03-20 20:48'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The TF.js abstraction has a fundamental architectural issue: tf-adapter.ts loads TF eagerly via require() at module import time, completely bypassing the async tf-backend.ts singleton. This means Clustering.init({ backend: 'wasm' }) has no effect on the actual TF instance used by algorithms. The browser adapter exports ~130 TF.js functions (including irrelevant conv/pooling ops), defeating tree-shaking. @tensorflow/tfjs-core is both a direct dep and included in @tensorflow/tfjs peer dep, causing duplication. React Native detection uses deprecated navigator.product. The tf-adapter-global.browser.ts is an orphaned test file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Single coherent TF.js loading path that respects Clustering.init backend configuration
- [x] #2 Browser adapter only exports the ~15 TF functions actually used by clustering algorithms
- [x] #3 @tensorflow/tfjs-core moved to peerDependencies to avoid duplication
- [x] #4 React Native detection uses robust modern approach (not deprecated navigator.product)
- [x] #5 Orphaned tf-adapter-global.browser.ts removed
- [x] #6 Webpack alias approach replaced with reliable module resolution
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add `ensureBackend()` to tf-backend.ts — synchronous auto-load for Node.js, race condition guard, global check for browser
2. Rewrite tf-adapter.ts with lazy named-export wrappers delegating to ensureBackend() (~33 functions)
3. Delete tf-adapter.browser.ts (universal adapter replaces it)
4. Delete orphaned tf-adapter-global.browser.ts
5. Remove webpack aliases from webpack.config.browser.js, add webpackIgnore to Node/RN dynamic imports
6. Replace deprecated navigator.product with multi-signal RN detection (HermesInternal, __fbBatchedBridge, nativeCallSyncHook)
7. Move @tensorflow/tfjs-core to peerDependencies + devDependencies, remove unused ml-matrix
8. Remove dead code (PlatformStorage, platformFetch, isWindows, isCI, getTensorFlow)
9. Add tests for platform detection, backend init, adapter wrappers, race condition
10. Run type-check, lint, and full test suite
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Unified TF.js backend architecture: lazy adapter wrappers, race condition guard, trimmed exports (~130→~33), multi-signal RN detection, removed dead code and orphaned files. All 206 tests pass, lint and type-check clean.
<!-- SECTION:NOTES:END -->
