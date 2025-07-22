---
id: task-20
title: Package and publish to npm
status: To Do
assignee: []
created_date: '2025-07-15'
labels: []
dependencies:
  - task-19
  - task-23
---

## Description

Prepare the library for publication to npm with proper packaging, versioning, and distribution setup for use in Node.js environments including VS Code extensions

## Acceptance Criteria

- [ ] Package.json properly configured for npm publication
- [ ] Build process generates CommonJS and ES modules
- [ ] TypeScript declaration files included in distribution
- [ ] LICENSE file added (appropriate open source license)
- [ ] CHANGELOG.md created with initial version
- [ ] npm publish scripts configured
- [ ] GitHub release automation setup
- [ ] Installation tested in VS Code extension environment
- [ ] Backend packaging strategy implemented:
  - [ ] Core package with CPU/WASM backends (no native dependencies)
  - [ ] Optional peer dependencies for tfjs-node and tfjs-node-gpu
  - [ ] Clear documentation on backend installation
  - [ ] Prebuilt binaries strategy evaluated
- [ ] Package size optimization (exclude unnecessary files)
- [ ] Browser and Node.js compatibility ensured

## Implementation Plan

### Backend Packaging Strategy

1. **Core Package Structure**:

   ```json
   {
     "name": "@clustering/core",
     "dependencies": {
       "@tensorflow/tfjs": "^4.x.x", // CPU and WASM backends included
       "ml-matrix": "^6.x.x"
     },
     "peerDependencies": {
       "@tensorflow/tfjs-node": "^4.x.x", // Optional for better Node.js performance
       "@tensorflow/tfjs-node-gpu": "^4.x.x" // Optional for CUDA support
     },
     "peerDependenciesMeta": {
       "@tensorflow/tfjs-node": { "optional": true },
       "@tensorflow/tfjs-node-gpu": { "optional": true }
     }
   }
   ```

2. **Installation Scenarios**:
   - **Browser/Basic Node.js**: `npm install @clustering/core` (uses CPU/WASM)
   - **Optimized Node.js**: `npm install @clustering/core @tensorflow/tfjs-node`
   - **GPU Accelerated**: `npm install @clustering/core @tensorflow/tfjs-node-gpu`

3. **Backend Auto-Detection**:

   ```typescript
   // Detect and use best available backend
   async function initializeBackend() {
     if (typeof window === 'undefined') {
       // Node.js environment
       try {
         await import('@tensorflow/tfjs-node-gpu');
         console.log('Using GPU backend');
       } catch {
         try {
           await import('@tensorflow/tfjs-node');
           console.log('Using native CPU backend');
         } catch {
           console.log('Using WASM backend');
         }
       }
     }
   }
   ```

4. **Prebuilt Binaries Consideration**:
   - TensorFlow.js handles prebuilt binaries for tfjs-node via @mapbox/node-pre-gyp
   - We don't need to manage native compilation ourselves
   - Users get prebuilt binaries for common platforms automatically
   - Fallback to source compilation only on unsupported platforms

5. **Documentation Requirements**:
   - Clear README section on backend options
   - Performance comparison table
   - Platform-specific installation guides
   - Troubleshooting guide for native dependency issues
