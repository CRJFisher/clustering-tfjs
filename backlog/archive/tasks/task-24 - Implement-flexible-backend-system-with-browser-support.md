---
id: task-24
title: Implement flexible backend system with browser support
status: Done
assignee:
  - '@chuck'
created_date: '2025-07-30'
updated_date: '2025-08-02'
labels: []
dependencies: []
---

## Description

Refactor the library to support dynamic backend selection, enabling automatic detection and use of the best available backend (CPU, WASM, WebGL, tfjs-node, tfjs-node-gpu) based on the environment and installed dependencies

## Acceptance Criteria

- [x] Dynamic backend detection and initialization system implemented
- [x] Browser support enabled with WebGL and WASM backends
- [x] Node.js backend auto-selection between tfjs-node and tfjs-node-gpu
- [x] Fallback mechanism when preferred backend unavailable
- [x] Backend can be manually specified via configuration
- [x] All algorithms work seamlessly across all backends
- [x] Performance maintained or improved across backends
- [x] Browser bundle created without Node.js dependencies (package.json configured)
- [x] Tests pass in both Node.js and browser environments
- [x] Documentation updated with backend configuration guide
- [x] README updated to reflect new installation options
- [x] API docs updated with backend configuration examples
- [x] Browser usage examples added to documentation
- [x] Migration guide for existing users updated
- [x] Multi-platform testing added to GitHub Actions

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

Completed full multi-platform support implementation across all 5 phases. Library now supports both browser (49KB bundle) and Node.js (163KB bundle) with automatic backend detection, optimized builds, comprehensive testing, and full documentation.
## Implementation Notes

### Research Findings (2024-01-30)

After investigating different approaches, here are the common patterns for handling multiple TensorFlow.js backends:

1. **Conditional Exports in package.json** - Modern approach using "browser" and "node" fields
2. **Dynamic imports with environment detection** - Runtime detection
3. **Separate entry points** - Different builds for browser vs Node.js
4. **Peer dependencies** - Let users choose which backend to install

### Challenges Encountered

1. **TypeScript namespace imports**: The `tf` namespace doesn't work well with dynamic imports
2. **Synchronous vs async initialization**: Current code expects synchronous access to tf
3. **Backward compatibility**: Need to maintain existing API while adding flexibility
4. **Build complexity**: Need separate builds for browser and Node.js

### Recommended Approach

Based on research, the best approach would be:

1. Create separate entry points for browser and Node.js
2. Use webpack or rollup to create browser bundles that import @tensorflow/tfjs
3. Keep Node.js builds using @tensorflow/tfjs-node with dynamic fallback
4. Use conditional exports in package.json to serve the right build

This is a significant architectural change that needs careful planning and might be better suited for a major version bump.

## Research Summary: Recommended Architecture

Based on comprehensive research of TensorFlow.js multi-platform patterns, the recommended approach is:

### 1. Platform Adapter Pattern
- Create `src/tf-adapter.ts` as a single import point for all TensorFlow operations
- Use build-time module aliasing to swap implementations (not runtime detection)
- Separate bundles for browser (`dist/clustering.browser.js`) and Node.js (`dist/clustering.node.js`)

### 2. Dependency Architecture
```json
{
  "dependencies": {
    "@tensorflow/tfjs-core": "^4.x.x"  // Only core as dependency
  },
  "peerDependencies": {
    "@tensorflow/tfjs": "^4.x.x",
    "@tensorflow/tfjs-backend-wasm": "^4.x.x",
    "@tensorflow/tfjs-node": "^4.x.x",
    "@tensorflow/tfjs-node-gpu": "^4.x.x"
  },
  "peerDependenciesMeta": {
    // All marked as optional
  }
}
```

### 3. Public API Design
```typescript
import { Clustering } from 'clustering-js';
import '@tensorflow/tfjs-backend-webgl';  // User imports their backend

await Clustering.init({ backend: 'webgl' });
const kmeans = new Clustering.KMeans({ k: 5 });
```

### 4. Build Configuration
- Use webpack `resolve.alias` or rollup `@rollup/plugin-alias`
- Create separate configs for browser and Node.js targets
- Configure package.json with `main`, `module`, and `browser` fields

### 5. TypeScript Strategy
- Use module augmentation for platform-specific types
- Conditional types for backend-aware API
- Maintain full type safety across platforms

## Updated Implementation Plan

Since the library is still in v0.1.x and hasn't been published to npm yet, this architectural change can be implemented as v0.2.0. Implementation phases:

### Phase 1: Decouple core logic (task 24.1)
- Remove hardcoded tfjs-node imports
- Create tf-adapter pattern
- Update dependencies

### Phase 2: Platform loaders and API (task 24.2)
- Create browser/node loader modules
- Implement async init() function
- Singleton pattern for tf instance

### Phase 3: Build system (task 24.3)
- Configure webpack/rollup
- Module aliasing for build-time resolution
- Separate browser/node bundles

### Phase 4: TypeScript enhancements (task 24.4)
- Module augmentation
- Conditional types
- Backend-aware type safety

### Phase 5: Testing and docs (task 24.5)
- Dual platform testing
- Migration guide
- Updated examples

## Decision

Given the complexity of this change:
1. Could be implemented for v0.2.0 since no users depend on the current API yet
2. Should still be implemented carefully in phases
3. Thoroughly tested on both platforms
4. Well-documented with clear examples

Alternative: Ship v0.1.0 as Node.js-only first to get feedback, then implement multi-platform support in v0.2.0.

## Final Implementation Summary (2025-08-02)

Successfully implemented complete multi-platform support across all 5 phases:

### Phase 1: Decoupled core logic from tfjs-node

- Created `tf-adapter.ts` as single import point
- Moved tfjs backends to peer dependencies
- Maintained backward compatibility

### Phase 2: Platform-specific loaders and public API

- Implemented `tf-loader.browser.ts` and `tf-loader.node.ts`
- Created singleton backend manager in `tf-backend.ts`
- Added `Clustering.init()` public API with platform detection
- Node.js loader has GPU → CPU → pure JS fallback chain

### Phase 3: Build system for multi-platform output

- Configured webpack for browser (49KB) and Node.js (163KB) bundles
- Set up module aliasing and conditional exports in package.json
- Created optimized builds with proper externals and tree-shaking

### Phase 4: Advanced TypeScript for type safety

- Added platform-aware types in `clustering-types.d.ts`
- Implemented conditional types for backend features
- Module augmentation for platform-specific properties

### Phase 5: Testing and documentation

- Created comprehensive migration guide (MIGRATION.md)
- Updated README with new initialization API
- Added browser (browser-webgl.html) and Node.js examples
- Built browser test suite (test/browser/index.html)
- Fixed all ESLint errors for clean build

### Additional work completed

- Added multi-platform testing to GitHub Actions
- Created dedicated `multi-platform.yml` workflow with:
  - Browser testing using Playwright
  - Node.js backend testing across multiple OS/versions
  - Bundle size checks (browser must stay under 100KB)
- Enhanced CI and release workflows with build verification

The library now fully supports both browser and Node.js environments with automatic backend detection, optimized bundles, and comprehensive testing.
