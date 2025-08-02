---
id: task-20
title: Package and publish to npm
status: Done
assignee: []
created_date: '2025-07-15'
updated_date: '2025-07-30'
labels: []
dependencies:
  - task-19
  - task-23
  - task-24
---

## Description

Prepare the library for publication to npm with proper packaging, versioning, and distribution setup for use in Node.js environments including VS Code extensions

## Acceptance Criteria

- [x] Package.json properly configured for npm publication
- [x] Build process generates CommonJS and ES modules
- [x] TypeScript declaration files included in distribution
- [x] LICENSE file added (appropriate open source license)
- [x] CHANGELOG.md created with initial version
- [x] npm publish scripts configured
- [x] GitHub release automation setup
- [x] Backend packaging strategy implemented:
  - [x] Core package with CPU/WASM backends (no native dependencies)
  - [x] Optional peer dependencies for tfjs-node and tfjs-node-gpu
  - [x] Prebuilt binaries strategy evaluated
- [x] Package size optimization (exclude unnecessary files)
- [ ] Set NPM_TOKEN in GitHub Actions secrets

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

3. **Prebuilt Binaries Consideration**:
   - TensorFlow.js handles prebuilt binaries for tfjs-node via @mapbox/node-pre-gyp
   - We don't need to manage native compilation ourselves
   - Users get prebuilt binaries for common platforms automatically
   - Fallback to source compilation only on unsupported platforms

4. **Documentation Requirements**:
   - Clear README section on backend options
   - Performance comparison table
   - Platform-specific installation guides
   - Troubleshooting guide for native dependency issues

## Implementation Notes

### Completed Steps (2024-01-30)

1. **Updated package.json for npm publication**:
   - Added comprehensive metadata (keywords, homepage, repository, bugs)
   - Configured dual module support with `exports` field
   - Moved @tensorflow/tfjs-node to optional peer dependencies
   - Added engines field requiring Node.js >= 18.0.0
   - Added npm scripts for versioning and releasing

2. **Created build configuration for CommonJS and ES modules**:
   - Created separate tsconfig files for CJS, ESM, and types
   - Implemented custom build script (scripts/build.js) to handle:
     - CommonJS output (dist/*.js)
     - ES module output (dist/*.esm.js)
     - TypeScript declarations (dist/*.d.ts)
   - Build process properly handles file renaming for ESM

3. **Added LICENSE file**:
   - MIT License with standard terms
   - Copyright attributed to "clustering-js contributors"

4. **Created CHANGELOG.md**:
   - Following Keep a Changelog format
   - Documented initial v0.1.0 release features
   - Listed all algorithms, validation metrics, and key features

5. **Configured npm scripts and .npmignore**:
   - Added comprehensive .npmignore to exclude development files
   - Added publishing scripts: release, release:minor, release:major
   - Added prepublishOnly hook to ensure quality checks
   - Added prepare script for automatic builds

6. **Set up GitHub release automation**:
   - Created .github/workflows/release.yml for automated npm publishing on tags
   - Created .github/workflows/ci.yml for continuous integration testing
   - CI tests on multiple OS (Ubuntu, Windows, macOS) and Node versions (18.x, 20.x)
   - Release workflow publishes to npm and creates GitHub releases

7. **Evaluated prebuilt binaries strategy**:
   - TensorFlow.js already handles prebuilt binaries through @mapbox/node-pre-gyp
   - No need for us to manage native compilation
   - Users automatically get prebuilt binaries for common platforms
   - Fallback to source compilation only on unsupported platforms

### Pending Work

- Test installation in a fresh project
- Backend documentation for README (installation guides, performance comparison)
- Create release script for automated workflow (task 20.1)

### Technical Decisions

1. **Module Strategy**: Dual CommonJS/ESM support using separate builds rather than trying to maintain compatibility in source
2. **Peer Dependencies**: Made tfjs-node and tfjs-node-gpu optional peers to avoid forcing native dependencies
3. **Build Process**: Custom Node.js script provides more control than pure TypeScript compiler
4. **Version Requirements**: Set Node.js >= 18 for modern JavaScript features and better TensorFlow.js support
