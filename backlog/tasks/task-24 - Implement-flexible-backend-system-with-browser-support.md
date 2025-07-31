---
id: task-24
title: Implement flexible backend system with browser support
status: To Do
assignee: []
created_date: '2025-07-30'
updated_date: '2025-07-30'
labels: []
dependencies: []
---

## Description

Refactor the library to support dynamic backend selection, enabling automatic detection and use of the best available backend (CPU, WASM, WebGL, tfjs-node, tfjs-node-gpu) based on the environment and installed dependencies

## Acceptance Criteria

- [ ] Dynamic backend detection and initialization system implemented
- [ ] Browser support enabled with WebGL and WASM backends
- [ ] Node.js backend auto-selection between tfjs-node and tfjs-node-gpu
- [ ] Fallback mechanism when preferred backend unavailable
- [ ] Backend can be manually specified via configuration
- [ ] All algorithms work seamlessly across all backends
- [ ] Performance maintained or improved across backends
- [x] Browser bundle created without Node.js dependencies (package.json configured)
- [ ] Tests pass in both Node.js and browser environments
- [ ] Documentation updated with backend configuration guide
- [x] README updated to reflect new installation options (partial - needs backend docs)
- [ ] API docs updated with backend configuration examples
- [ ] Browser usage examples added to documentation
- [ ] Migration guide for existing users updated

## Implementation Notes

### Work completed in task 20 (2024-01-30)

1. **Package structure prepared for flexible backends**:
   - Moved @tensorflow/tfjs-node and tfjs-node-gpu to optional peer dependencies
   - Core package depends only on @tensorflow/tfjs (includes CPU/WASM backends)
   - Configured dual module support (CommonJS and ESM) for browser compatibility
   - **Note**: The package.json shows installation scenarios like `@clustering/core` but the package is actually named `clustering-js`
   - **Important**: Despite this setup, the code still hard-imports `@tensorflow/tfjs-node` everywhere, so none of this flexibility actually works yet

2. **Build system ready for browser support**:
   - Created separate build configurations for CJS and ESM
   - Package exports configured for modern bundlers
   - .npmignore optimized to exclude Node-only files from browser bundles

### Remaining work

The main remaining work is to replace all hard-coded imports of `@tensorflow/tfjs-node` with dynamic imports that detect the environment and available backends. Currently, all source files import tfjs-node directly, which prevents browser usage.

## Implementation Plan

### Backend Auto-Detection Strategy

From the original task 20 planning:

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

### Implementation Steps

1. Create a central backend manager module
2. Replace all `import * as tf from "@tensorflow/tfjs-node"` with dynamic imports
3. Ensure all algorithms initialize the backend before use
4. Add configuration options for manual backend selection
5. Update build process to create browser-compatible bundles
6. Add browser-specific tests
7. Update documentation with backend usage guide
