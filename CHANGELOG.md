# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-03-20

### Added

- React Native backend support with environment detection, platform-specific loader, and I/O compatibility
- `SOM.cluster()` method for two-phase SOM clustering with meaningful cluster output
- Lanczos iterative eigensolver for scalable spectral clustering (49.8x speedup at n=500, enables n=5000+)
- Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) validation metrics
- Elbow method and silhouette-only evaluation for cluster selection
- `partialFit()` dimension validation for SOM
- `dispose()` methods for proper tensor cleanup across all algorithms
- Comprehensive test coverage improvements (~1800 lines of new tests)

### Changed

- Agglomerative clustering optimized from O(n³) to O(n² log n)
- Unified TF.js backend architecture with lazy adapter wrappers
- Reduced public API surface from ~130 to ~33 exports
- Moved `@tensorflow/tfjs-core` to peerDependencies
- `SOM.getWeights()` returns plain arrays instead of tensors
- `Clustering.init()` is now idempotent with promise-based concurrency guard
- Build output restructured to flat `dist/` layout with `sideEffects: false` for tree-shaking

### Fixed

- Spectral embedding normalization using D^(1/2) instead of D across all four spectral paths
- Tensor memory leaks across KMeans, SpectralClustering, SOM, and validation functions
- 6 SOM algorithmic correctness issues (weight update normalization, 8-connectivity, iterative BMU, epoch shuffling, PCA initialization, density map convolution)
- 6 eigendecomposition numerical stability issues (division by zero, PSD clamping, variable shadowing, Wilkinson shift NaN, tolerance, eigenvector orthogonality)
- Broken `package.json` exports and build output structure; test files no longer shipped to npm
- `findOptimalClusters` combined scoring and NaN handling
- Platform detection reliability with multi-signal approach

## [0.4.0] - 2025-09-03

### Added

- Self-Organizing Maps (SOM) algorithm with rectangular and hexagonal grid topologies
- Multiple SOM neighborhood functions (gaussian, bubble, mexican_hat)
- Three SOM initialization methods (random, linear, PCA)
- Online/incremental learning support via `partialFit()` and streaming mode
- SOM visualization utilities (component planes, hit maps, activation maps, U-matrix)
- Model persistence with `saveToJSON()` / `loadFromJSON()`
- SOM quality metrics (quantization error, topographic error)

## [0.3.1] - 2025-08-03

### Fixed

- Browser tensor2d compatibility issues
- CI failures on Ubuntu and Windows platforms
- WASM backend test initialization
- Export of tensorflow-helper module for cross-platform compatibility
- Windows tfjs-node fallback handling

### Added

- Backend matrix CI workflow
- Playwright configuration for browser testing

## [0.3.0] - 2025-08-02

### Changed

- Build system configured for multi-platform output (browser + Node.js bundles)
- CI/release workflow updates for multi-platform builds

## [0.2.0] - 2025-08-02

### Added

- Multi-platform support (browser + Node.js)
- Platform-specific loaders and automatic backend detection
- Clustering namespace and public API (`Clustering.init()`)
- Platform-optimized bundles (browser and Node.js)
- Advanced TypeScript type safety
- Migration guide (MIGRATION.md)

### Changed

- Package renamed from `clustering-js` to `clustering-tfjs`
- Decoupled core logic from `tfjs-node` for browser compatibility

## [0.1.0] - 2024-01-30

### Added

- Initial release of clustering-tfjs
- K-Means clustering algorithm with K-means++ initialization
- Spectral Clustering with RBF and k-NN affinity options
- Agglomerative Clustering with multiple linkage methods (ward, complete, average, single)
- Validation metrics:
  - Silhouette Score
  - Davies-Bouldin Index
  - Calinski-Harabasz Index
- `findOptimalClusters` utility for automatic cluster number selection
- Full TypeScript support with comprehensive type definitions
- Support for both array and TensorFlow.js tensor inputs
- Scikit-learn compatible API and results
- Optional TensorFlow.js backend acceleration (CPU, WASM, tfjs-node, tfjs-node-gpu)
- Comprehensive test suite with sklearn parity tests
- Benchmarking framework for performance monitoring
- Examples and documentation

### Technical Details

- Built with TensorFlow.js for high-performance numerical computations
- Memory-efficient implementations with proper tensor disposal
- Browser and Node.js compatibility
- CommonJS and ES module builds
- Zero required native dependencies (optional tfjs-node for better performance)

[0.5.0]: https://github.com/CRJFisher/clustering-tfjs/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/CRJFisher/clustering-tfjs/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/CRJFisher/clustering-tfjs/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/CRJFisher/clustering-tfjs/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/CRJFisher/clustering-tfjs/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/CRJFisher/clustering-tfjs/releases/tag/v0.1.0