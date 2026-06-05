# Migration Guide: snake_case API

The public API uses Python-style `snake_case` for all function names, method
names, and option keys, matching scikit-learn. Class, interface, and type names
stay `PascalCase`.

## What changed

Rename option keys and methods at every call site:

| Old (camelCase) | New (snake_case) |
| --- | --- |
| `nClusters` | `n_clusters` |
| `randomState` | `random_state` |
| `nInit` | `n_init` |
| `maxIter` | `max_iter` |
| `nNeighbors` | `n_neighbors` |
| `gridWidth` / `gridHeight` | `grid_width` / `grid_height` |
| `numEpochs` | `num_epochs` |
| `learningRate` | `learning_rate` |
| `miniBatchSize` | `mini_batch_size` |
| `onlineMode` | `online_mode` |
| `model.fitPredict(X)` | `model.fit_predict(X)` |
| `findOptimalClusters(...)` | `find_optimal_clusters(...)` |
| `computeWss(...)` | `compute_wss(...)` |
| `silhouetteScore`, `daviesBouldin`, `calinskiHarabasz` | `silhouette_score`, `davies_bouldin`, `calinski_harabasz` |
| `adjustedRandIndex`, `normalizedMutualInfo` | `adjusted_rand_index`, `normalized_mutual_info` |

```typescript
// Before
const km = new KMeans({ nClusters: 3, randomState: 42, nInit: 10 });
const labels = await km.fitPredict(X);

// After
const km = new KMeans({ n_clusters: 3, random_state: 42, n_init: 10 });
const labels = await km.fit_predict(X);
```

The `clustering-tfjs/utils` subpath export is removed; import its members
(`pairwise_distance_matrix`, `find_optimal_clusters`, `compute_wss`,
`find_knee`) from the package root instead.

---

# Migration Guide: Multi-Platform Support

This guide helps you migrate to the new multi-platform version of clustering-tfjs.

## What's New

The library now supports both browser and Node.js environments with automatic backend detection and optimized bundles for each platform.

### Key Changes

1. **New Initialization API**: The library now requires initialization before use
2. **Platform-Specific Bundles**: Separate optimized bundles for browser and Node.js
3. **Flexible Backend Selection**: Choose between CPU, WebGL, WASM, and Node.js backends
4. **TypeScript Improvements**: Platform-aware types and better type safety

## Migration Steps

### 1. Update Your Imports

The library exports remain mostly the same, but you now have access to the `Clustering` namespace:

```typescript
// Old way - still works
import { KMeans, SpectralClustering } from 'clustering-tfjs';

// New way - recommended
import { Clustering } from 'clustering-tfjs';
```

### 2. Initialize Before Use (Browser)

In browser environments, you must initialize the library:

```typescript
// Make sure TensorFlow.js is loaded first
// <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/dist/tf.min.js"></script>

import { Clustering } from 'clustering-tfjs';

// Initialize with auto-detection
await Clustering.init();

// Or specify a backend
await Clustering.init({ backend: 'webgl' });

// Now use the algorithms
const kmeans = new Clustering.KMeans({ n_clusters: 3 });
```

### 3. Node.js Usage

In Node.js, the library works similarly but with automatic backend selection:

```typescript
import { Clustering } from 'clustering-tfjs';

// Optional: Initialize to use GPU backend if available
await Clustering.init();

// Use algorithms
const kmeans = new Clustering.KMeans({ n_clusters: 3 });
const labels = await kmeans.fit_predict(data);
```

### 4. Platform Detection

You can now detect the current platform and available features:

```typescript
console.log('Platform:', Clustering.platform); // 'browser' or 'node'
console.log('Features:', Clustering.features);
// {
//   gpu_acceleration: boolean,
//   wasm_simd: boolean,
//   node_bindings: boolean,
//   webgl: boolean
// }
```

## Backend Options

### Browser Backends

- `'cpu'` - Pure JavaScript (slowest, most compatible)
- `'webgl'` - WebGL acceleration (recommended for browsers)
- `'wasm'` - WebAssembly backend

### Node.js Backends

- `'cpu'` - Pure JavaScript
- `'tensorflow'` - Native TensorFlow C++ bindings (fastest)

## Installation Changes

### Browser

```html
<!-- Load TensorFlow.js first -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/dist/tf.min.js"></script>

<!-- Then load clustering-tfjs -->
<script src="https://unpkg.com/clustering-tfjs/dist/clustering.browser.js"></script>
```

### Node.js

```bash
# Basic installation (CPU only)
npm install clustering-tfjs

# With Node.js acceleration
npm install clustering-tfjs @tensorflow/tfjs-node

# With GPU support
npm install clustering-tfjs @tensorflow/tfjs-node-gpu
```

## Breaking Changes

1. **Initialization Required**: You must call `Clustering.init()` before using algorithms in the browser
2. **Async API**: All clustering methods now return Promises
3. **Bundle Size**: Browser bundle no longer includes Node.js dependencies

## Examples

### Basic Browser Example

```html
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/dist/tf.min.js"></script>
    <script src="https://unpkg.com/clustering-tfjs/dist/clustering.browser.js"></script>
</head>
<body>
    <script>
        async function cluster() {
            // Initialize
            await ClusteringTFJS.Clustering.init({ backend: 'webgl' });
            
            // Prepare data
            const data = [[0, 0], [0, 1], [5, 5], [5, 6]];
            
            // Cluster
            const kmeans = new ClusteringTFJS.KMeans({ n_clusters: 2 });
            const labels = await kmeans.fit_predict(data);
            
            console.log('Cluster labels:', labels);
        }
        
        cluster();
    </script>
</body>
</html>
```

### Node.js Example

```javascript
const { Clustering } = require('clustering-tfjs');

async function main() {
    // Optional initialization
    await Clustering.init();
    
    // Your data
    const data = [
        [0, 0], [0, 1], [1, 0],
        [5, 5], [5, 6], [6, 5]
    ];
    
    // K-Means clustering
    const kmeans = new Clustering.KMeans({ n_clusters: 2 });
    const labels = await kmeans.fit_predict(data);
    console.log('K-Means labels:', labels);
    
    // Spectral clustering
    const spectral = new Clustering.SpectralClustering({ 
        n_clusters: 2,
        affinity: 'rbf'
    });
    const spectral_labels = await spectral.fit_predict(data);
    console.log('Spectral labels:', spectral_labels);
}

main();
```

## Troubleshooting

### "TensorFlow.js not initialized" Error

Make sure to call `Clustering.init()` before using any algorithms:

```typescript
// ❌ Wrong
const kmeans = new Clustering.KMeans({ n_clusters: 3 });

// ✅ Correct
await Clustering.init();
const kmeans = new Clustering.KMeans({ n_clusters: 3 });
```

### Performance Issues

Check which backend is being used:

```typescript
await Clustering.init();
console.log('Current backend:', await tf.getBackend());
```

For best performance:
- Browser: Use WebGL backend
- Node.js: Install @tensorflow/tfjs-node or @tensorflow/tfjs-node-gpu

### Bundle Size Concerns

The browser bundle is now much smaller (49KB) as it doesn't include Node.js dependencies. TensorFlow.js must be loaded separately.

## Need Help?

- Check the [examples](./examples) directory for more usage patterns
- See the [API documentation](./docs/api.md) for detailed reference
- Report issues on [GitHub](https://github.com/CRJFisher/clustering-tfjs/issues)