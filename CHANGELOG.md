# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/yourusername/clustering-tfjs/releases/tag/v0.1.0