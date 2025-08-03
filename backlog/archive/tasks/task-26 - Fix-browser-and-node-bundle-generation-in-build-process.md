---
id: task-26
title: Fix browser and node bundle generation in build process
status: Done
assignee:
  - '@assistant'
created_date: '2025-08-03'
updated_date: '2025-08-03'
labels: []
dependencies: []
---

## Description

The build process is only creating dist/index.js but not dist/clustering.browser.js and dist/clustering.node.js, causing size-check and other tests to fail

## Acceptance Criteria

- [x] dist/clustering.browser.js is created by build process
- [x] dist/clustering.node.js is created by build process
- [x] size-check job passes
- [x] All bundle files have appropriate sizes

## Implementation Plan

1. Investigate why browser and node bundles are not being created
2. Check build scripts and configurations
3. Update CI workflow to use correct build command
4. Verify bundle sizes are within limits

## Implementation Notes

### Root Cause

The size-check job in `.github/workflows/multi-platform.yml` was using `npm run build` which only runs the basic `scripts/build.js`. This script only creates:

- `dist/index.js` (CommonJS)
- `dist/index.esm.js` (ES Module)
- TypeScript declarations

The browser and node bundles are created by `scripts/build-multi-platform.js` which is invoked by `npm run build:multi`.

### Solution

Changed line 194 in `.github/workflows/multi-platform.yml`:

```yaml
- name: Build all targets
  run: npm run build:multi  # Changed from: npm run build
```

### Bundle Sizes

After the fix, the bundles are successfully created with these sizes:

- Browser bundle: 65K (well under the 100K limit)
- Node.js bundle: 179K
- Standard build: 4.7K

### Build Scripts Overview

- `npm run build` - Basic build (CommonJS + ESM + types)
- `npm run build:browser` - Browser bundle only
- `npm run build:node` - Node.js bundle only
- `npm run build:multi` - All bundles (includes webpack builds)

### Related Files

- `.github/workflows/multi-platform.yml` - Contains the size-check job
- `scripts/build.js` - Basic build script
- `scripts/build-multi-platform.js` - Multi-platform build script
- `webpack.config.browser.js` - Browser bundle configuration
- `webpack.config.node.js` - Node.js bundle configuration
