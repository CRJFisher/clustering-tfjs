---
id: DRAFT-1
title: Add React Native backend support
status: To Do
assignee: []
created_date: '2025-09-02 14:26'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

Enable clustering-js to run in React Native environments using the TensorFlow.js React Native backend. This will allow mobile applications to use the clustering library with GPU acceleration via the rn-webgl backend.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [ ] #1 Library detects React Native environment correctly
- [ ] #2 rn-webgl backend initializes successfully with fallback to CPU
- [ ] #3 All clustering algorithms work in React Native
- [ ] #4 Performance optimizations for mobile are implemented
- [ ] #5 Documentation includes React Native setup guide
- [ ] #6 Example React Native app demonstrates usage
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->

1. **Update environment detection in `src/tf-backend.ts`**
   - Add React Native detection using `navigator.product === 'ReactNative'`
   - Update `loadBackend()` to handle RN case alongside Node and browser

2. **Create React Native loader module**
   - Add `src/tf-loader.rn.ts` similar to existing browser/node loaders
   - Import `@tensorflow/tfjs-react-native` to register the backend
   - Set backend to `rn-webgl` with CPU fallback

3. **Update type definitions**
   - Add 'react-native' to Platform type in `clustering-types.ts`
   - Add 'rn-webgl' to backend options
   - Update BackendConfig interface

4. **Handle React Native I/O constraints**
   - No filesystem access - rely on bundled resources, HTTP fetch, or AsyncStorage
   - Update any file-based operations to check platform first

5. **Implement performance optimizations**
   - Use float32 tensors by default
   - Add tensor reuse patterns with tf.tidy
   - Implement warmup function for first-run graph compilation

6. **Add React Native example**
   - Create example app (Expo or bare RN)
   - Demonstrate clustering with visualization
   - Include performance monitoring

7. **Update documentation**
   - Add React Native section to README
   - Include package dependencies (@tensorflow/tfjs-react-native, GL bindings)
   - Document Expo vs bare React Native setup

## Technical Notes

- **Backend ID**: Use `rn-webgl` (not webgpu which doesn't work in RN)
- **Required packages**: `@tensorflow/tfjs`, `@tensorflow/tfjs-react-native`, GL bindings (via Expo or manual setup)
- **Initialization**: Must call `await tf.ready()` before `setBackend()`
- **No WebGPU**: Existing webgpu backend won't work in RN environment
<!-- SECTION:PLAN:END -->
