# Windows Compatibility Guide

## TensorFlow.js Backend Selection

This library uses TensorFlow.js for high-performance numerical computations. On different platforms, you may need different TensorFlow.js backends:

### Recommended Setup by Platform

#### Linux/macOS
```bash
npm install clustering-tfjs @tensorflow/tfjs-node
```

#### Windows
Due to native binding compilation issues on Windows, we recommend using the pure JavaScript backend:

```bash
npm install clustering-tfjs @tensorflow/tfjs
```

If you encounter errors like "The specified module could not be found" or "tfjs_binding.node", this is because `@tensorflow/tfjs-node` requires native bindings that may fail to compile on Windows.

### Modifying Imports for Windows

If you've already installed the library with `@tensorflow/tfjs-node`, you can work around Windows issues by:

1. **Option 1: Replace the dependency**
   ```bash
   npm uninstall @tensorflow/tfjs-node
   npm install @tensorflow/tfjs
   ```

2. **Option 2: Use a bundler with alias** (webpack, vite, etc.)
   ```javascript
   // webpack.config.js
   module.exports = {
     resolve: {
       alias: {
         '@tensorflow/tfjs-node': '@tensorflow/tfjs'
       }
     }
   };
   ```

3. **Option 3: Use module-alias in your application**
   ```bash
   npm install module-alias
   ```
   
   Then in your application's entry point:
   ```javascript
   require('module-alias').addAlias('@tensorflow/tfjs-node', '@tensorflow/tfjs');
   ```

### Performance Considerations

- `@tensorflow/tfjs-node`: Uses native C++ bindings for optimal performance (2-10x faster)
- `@tensorflow/tfjs`: Pure JavaScript implementation, slower but works everywhere
- `@tensorflow/tfjs-node-gpu`: GPU acceleration for Linux systems with CUDA

For most clustering tasks, the pure JavaScript backend performance is acceptable. The native backend is recommended for:
- Large datasets (>10,000 points)
- Real-time applications
- Repeated clustering operations

### CI/CD Considerations

For CI environments, especially Windows runners, we recommend:

1. Setting environment variables to suppress TensorFlow warnings:
   ```yaml
   env:
     TF_CPP_MIN_LOG_LEVEL: '3'
   ```

2. Using the test setup that automatically falls back to `@tensorflow/tfjs`:
   ```javascript
   // jest.config.js
   setupFilesAfterEnv: ['<rootDir>/test/setup.js']
   ```

### Troubleshooting

If you see errors like:
- "Error: The specified module could not be found"
- "Cannot find module './tfjs_binding.node'"
- "Error loading shared library"

These indicate `@tensorflow/tfjs-node` native bindings failed to load. Use one of the solutions above to switch to the pure JavaScript backend.