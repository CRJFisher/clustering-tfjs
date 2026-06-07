---
id: DRAFT-3
title: Implement React Native detection and loader module
status: To Do
assignee: []
created_date: '2025-09-03 21:37'
updated_date: '2025-09-03 21:44'
labels: []
dependencies: []
parent_task_id: TASK-32
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

Add React Native environment detection and create the loader module to properly initialize the TensorFlow.js React Native backend with appropriate fallbacks.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [x] #1 React Native environment correctly detected via navigator.product check
- [x] #2 New tf-loader.rn.ts module created and functional
- [x] #3 rn-webgl backend registers and initializes
- [x] #4 CPU fallback works when GPU unavailable
- [x] #5 tf.ready() called before backend initialization
- [x] #6 Backend switching logic updated in tf-backend.ts
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->

1. Create src/tf-loader.rn.ts module similar to existing loaders
2. Import @tensorflow/tfjs-react-native package
3. Add React Native detection logic using navigator.product === 'ReactNative'
4. Implement async backend initialization with tf.ready()
5. Configure rn-webgl as primary backend with CPU fallback
6. Update tf-backend.ts to include RN loader in platform detection
7. Add proper error handling and logging for RN initialization
8. Test backend switching between rn-webgl and CPU
9. Verify tensor operations work correctly after initialization
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->

### Created Files

- **src/tf-loader.rn.ts**: New React Native loader module with WebGL/CPU fallback

### Modified Files

- **src/tf-backend.ts**:
  - Added React Native detection via `navigator.product === 'ReactNative'`
  - Updated loadBackend() to prioritize RN check before Node.js
  - Added 'rn-webgl' to BackendConfig options documentation

### Key Implementation Details

1. **Environment Detection**:
   - React Native detected first using `navigator.product`
   - Prevents false positive with Node.js detection in RN environment
   - Maintains backward compatibility with existing browser/Node detection

2. **Backend Loading**:
   - Follows existing loader pattern for consistency
   - Implements try-catch fallback from rn-webgl to CPU
   - Calls tf.ready() before backend initialization per RN requirements

3. **Error Handling**:
   - Clear error messages with installation instructions
   - Separate instructions for Expo vs bare React Native
   - Console logging for successful backend initialization

4. **Backend Configuration**:
   - Added 'rn-webgl' as a valid backend option
   - Maintains auto-detection as primary approach
   - Supports manual backend override via config
   <!-- SECTION:NOTES:END -->
