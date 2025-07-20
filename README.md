# clustering-js

🚧 **Work in Progress** 🚧

A TypeScript implementation of clustering algorithms using TensorFlow.js, aiming for compatibility with scikit-learn.

## Status

This library is currently under active development. The goal is to provide native clustering algorithms in TypeScript/JavaScript that produce results compatible with scikit-learn's implementations.

### Currently Implemented

- ✅ Core tensor utilities
- ✅ Base clustering interfaces and types
- 🔄 **AgglomerativeClustering** (partial implementation)
- 🔄 **SpectralClustering** (working toward sklearn parity)
- 🔄 **K-Means** (with deterministic seeding)

### In Progress

- Achieving full sklearn compatibility for SpectralClustering
- Comprehensive validation metrics (Calinski-Harabasz, Davies-Bouldin, Silhouette)
- Performance optimizations and memory efficiency

## Installation

```bash
npm install clustering-js
```

## Usage

⚠️ **Note**: API is subject to change during development.

```typescript
import { SpectralClustering, KMeans } from 'clustering-js';

// Basic usage (API may change)
const spectral = new SpectralClustering({ nClusters: 3 });
// Implementation in progress...
```

## Development

This project uses a task-based development approach with detailed tracking in the `backlog/` directory.

```bash
# Install dependencies
npm install

# Run tests
npm test

# Build
npm run build
```

## Contributing

This is an active research and development project. The codebase is evolving rapidly as we work toward sklearn compatibility.

## License

MIT

---

**Roadmap**: The ultimate goal is to provide a comprehensive, performant clustering library for JavaScript/TypeScript environments that matches scikit-learn's behavior and can be used in both Node.js and browser environments.
