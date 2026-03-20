---
id: task-43
title: Fix TensorFlow.js backend architecture and platform detection
status: To Do
assignee: []
created_date: '2026-03-20'
labels: []
dependencies: []
---

## Description

The TF.js abstraction has a fundamental architectural issue: tf-adapter.ts loads TF eagerly via require() at module import time, completely bypassing the async tf-backend.ts singleton. This means Clustering.init({ backend: 'wasm' }) has no effect on the actual TF instance used by algorithms. The browser adapter exports ~130 TF.js functions (including irrelevant conv/pooling ops), defeating tree-shaking. @tensorflow/tfjs-core is both a direct dep and included in @tensorflow/tfjs peer dep, causing duplication. React Native detection uses deprecated navigator.product. The tf-adapter-global.browser.ts is an orphaned test file.

## Acceptance Criteria

- [ ] Single coherent TF.js loading path that respects Clustering.init backend configuration
- [ ] Browser adapter only exports the ~15 TF functions actually used by clustering algorithms
- [ ] @tensorflow/tfjs-core moved to peerDependencies to avoid duplication
- [ ] React Native detection uses robust modern approach (not deprecated navigator.product)
- [ ] Orphaned tf-adapter-global.browser.ts removed
- [ ] Webpack alias approach replaced with reliable module resolution
