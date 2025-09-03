---
id: task-32.2
title: Implement React Native detection and loader module
status: To Do
assignee: []
created_date: '2025-09-03 21:37'
updated_date: '2025-09-03 21:44'
labels: []
dependencies: []
parent_task_id: task-32
---

## Description

Add React Native environment detection and create the loader module to properly initialize the TensorFlow.js React Native backend with appropriate fallbacks.

## Acceptance Criteria

- [ ] React Native environment correctly detected via navigator.product check
- [ ] New tf-loader.rn.ts module created and functional
- [ ] rn-webgl backend registers and initializes
- [ ] CPU fallback works when GPU unavailable
- [ ] tf.ready() called before backend initialization
- [ ] Backend switching logic updated in tf-backend.ts

## Implementation Plan

1. Create src/tf-loader.rn.ts module similar to existing loaders
2. Import @tensorflow/tfjs-react-native package
3. Add React Native detection logic using navigator.product === 'ReactNative'
4. Implement async backend initialization with tf.ready()
5. Configure rn-webgl as primary backend with CPU fallback
6. Update tf-backend.ts to include RN loader in platform detection
7. Add proper error handling and logging for RN initialization
8. Test backend switching between rn-webgl and CPU
9. Verify tensor operations work correctly after initialization
