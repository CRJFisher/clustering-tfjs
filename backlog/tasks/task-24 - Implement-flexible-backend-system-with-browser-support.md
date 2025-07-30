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

2. **Build system ready for browser support**:
   - Created separate build configurations for CJS and ESM
   - Package exports configured for modern bundlers
   - .npmignore optimized to exclude Node-only files from browser bundles

### Remaining work

The main remaining work is to replace all hard-coded imports of `@tensorflow/tfjs-node` with dynamic imports that detect the environment and available backends. Currently, all source files import tfjs-node directly, which prevents browser usage
