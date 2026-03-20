# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.4.0]: https://github.com/CRJFisher/clustering-tfjs/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/CRJFisher/clustering-tfjs/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/CRJFisher/clustering-tfjs/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/CRJFisher/clustering-tfjs/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/CRJFisher/clustering-tfjs/releases/tag/v0.1.0