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

- [x] Windows tests pass without backend errors
- [x] Tests can use appropriate TensorFlow backends on Windows
- [x] No regression in test functionality

## Implementation Plan

1. Investigate why Jest moduleNameMapper isn't working
2. Find alternative approach to handle tfjs-node on Windows CI
3. Update test configuration to use appropriate backends
4. Ensure all tests pass on Windows

## Implementation Notes

### Root Cause

Windows CI environments have issues loading native Node.js modules. The `@tensorflow/tfjs-node` package includes native bindings (`tfjs_binding.node`) that fail to load with:
```
The specified module could not be found.
\\?\D:\a\clustering-tfjs\clustering-tfjs\node_modules\@tensorflow\tfjs-node\lib\napi-v8\tfjs_binding.node
```

Additionally, the test `findOptimalClusters.test.ts` was hardcoded to use the "tensorflow" backend, which only exists in `@tensorflow/tfjs-node`, not in the pure JavaScript `@tensorflow/tfjs`.

### Initial Approach (Failed)

Tried using Jest's `moduleNameMapper` to redirect imports:
```javascript
moduleNameMapper['^@tensorflow/tfjs-node$'] = '@tensorflow/tfjs';
```

This approach failed because:
1. The mapping happens too late in the import process
2. It doesn't handle the `export * from` statements properly
3. Type definitions still reference tfjs-node

### Final Solution

Implemented a multi-layered approach:

1. **Updated `src/tf-adapter.ts`** to dynamically load TensorFlow.js:
   ```typescript
   if (process.platform === 'win32' && process.env.CI) {
     tf = require('@tensorflow/tfjs');
   } else {
     try {
       tf = require('@tensorflow/tfjs-node');
     } catch (error) {
       tf = require('@tensorflow/tfjs');
     }
   }
   ```

2. **Modified `test/setup.js`** to set environment variable:
   ```javascript
   if (process.platform === 'win32' && process.env.CI) {
     process.env.TF_FORCE_CPU_BACKEND = 'true';
   }
   ```

3. **Updated `test/tensorflow-helper.ts`** to respect the environment variable:
   ```typescript
   if (process.env.TF_FORCE_CPU_BACKEND === 'true') {
     tf = require('@tensorflow/tfjs');
   } else {
     try {
       tf = require('@tensorflow/tfjs-node');
     } catch (error) {
       tf = require('@tensorflow/tfjs');
     }
   }
   ```

4. **Fixed `test/utils/findOptimalClusters.test.ts`** to detect backends dynamically:
   ```typescript
   const backends = tf.engine().registryFactory;
   if ('tensorflow' in backends) {
     await tf.setBackend("tensorflow");
   } else {
     await tf.setBackend("cpu");
   }
   ```

### Benefits

- Windows CI uses pure JavaScript implementation (CPU backend)
- Other platforms continue to use optimized native backend
- Graceful fallback if tfjs-node fails to load on any platform
- No hardcoded backend assumptions in tests
- Type safety maintained

### Related Files

- `src/tf-adapter.ts` - Main TensorFlow.js adapter
- `test/setup.js` - Jest setup file
- `test/tensorflow-helper.ts` - Test-specific TF helper
- `test/utils/findOptimalClusters.test.ts` - Test with backend selection
- `jest.config.js` - Jest configuration (moduleNameMapper removed)
