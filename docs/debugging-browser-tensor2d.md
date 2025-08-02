# Debugging Browser tensor2d Error

## Problem Statement
In browser backend tests, we're getting `TypeError: o.tensor2d is not a function` even though:
- TensorFlow is loaded and ready (backend: webgl)
- `tf.tensor2d` is available as a function
- `window.tf.tensor2d` is available as a function

## Error Details
```
Browser: tf.tensor2d available? function
Browser: window.tf.tensor2d available? function
Browser: Initialized backend: webgl
Browser: Creating KMeans instance...
Browser: Fitting KMeans with data: JSHandle@array
Browser: Test failed: TypeError: o.tensor2d is not a function
    at u.fit (http://localhost:42551/dist/clustering.browser.js:1:23108)
```

## Analysis

### What We Know
1. The global `tf` object exists and has `tensor2d` as a function
2. The error happens inside the minified code (`o.tensor2d`)
3. The variable `o` is what our code thinks is the tf namespace
4. This suggests a module resolution or import issue

### Current Setup
1. **Webpack Config**: `@tensorflow/tfjs` is marked as external with `root: 'tf'`
2. **tf-adapter.browser.ts**: Imports `@tensorflow/tfjs` which webpack should map to global `tf`
3. **All files**: Use `import * as tf from '../tf-adapter'`

### Theories

#### Theory 1: Webpack Module Resolution Issue
- Webpack might be treating the tf import differently than expected
- The minified `o` variable might not be properly linked to the global `tf`

#### Theory 2: Export/Import Mismatch
- The tf-adapter.browser.ts exports both default and namespace
- There might be a mismatch in how webpack handles this

#### Theory 3: Browser Bundle Not Including tf-adapter.browser
- The build might not be using the browser-specific adapter
- Need to verify webpack alias is working

## Debugging Steps Attempted

### Attempt 1: Changed Import Style
- Changed all `import tf from` to `import * as tf from`
- Result: Still failing with same error

### Attempt 2: Simplified tf-adapter.browser.ts
- Directly import and re-export from '@tensorflow/tfjs'
- Result: Still failing

### Attempt 3: Added Debug Logging
- Confirmed tf.tensor2d exists in browser
- Result: Function exists but still fails when called from within the bundle

### Attempt 4: Added Logging to tf-adapter.browser.ts
- Added console.log to see what tfImport contains
- This will help us understand if webpack is properly resolving the external

### Attempt 5: Direct Global Access with Delegation
- Created a namespace object that uses getters to delegate to window.tf
- This ensures we're always using the global tf, not a webpack module
- Explicitly exports all needed functions

## Next Steps to Try

1. **Examine the minified code** to see what `o` actually is
2. **Add console.log in tf-adapter.browser.ts** to verify it's being used
3. **Check webpack bundle** to see how imports are being resolved
4. **Try using tf directly** without going through adapter
5. **Create minimal reproduction** outside of test framework

## Potential Solutions

1. **Direct Global Access**: Instead of import, use `(window as any).tf` 
2. **Different Webpack Config**: Change how externals are configured
3. **Runtime Binding**: Bind tf functions at runtime instead of import time
4. **Build Process Change**: Use different bundler or config for browser

## WASM Backend Specific Issues

### Additional WASM Observations
- Backend successfully switches from webgl to wasm
- Same `o.tensor2d is not a function` error
- This confirms the issue is not backend-specific but related to module resolution

### WASM Test Output
```
Browser: TensorFlow ready, current backend: webgl
Browser: tf.tensor2d available? function
Browser: window.tf.tensor2d available? function
Browser: Initialized backend: wasm
Browser: Creating KMeans instance...
Browser: Fitting KMeans with data: JSHandle@array
Browser: Test failed: TypeError: o.tensor2d is not a function
```

### Key Insight
The fact that both WebGL and WASM backends fail with the exact same error at the same location (`http://localhost:40279/dist/clustering.browser.js:1:23108`) strongly suggests this is a **build/bundling issue**, not a runtime backend issue.