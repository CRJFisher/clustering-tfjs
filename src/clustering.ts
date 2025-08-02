/**
 * Main entry point for the clustering library
 * 
 * Provides initialization and configuration for multi-platform support.
 */

import { initializeBackend, BackendConfig } from './tf-backend';
import { KMeans } from './clustering/kmeans';
import { SpectralClustering } from './clustering/spectral';
import { AgglomerativeClustering } from './clustering/agglomerative';

// Re-export all clustering algorithms and utilities
export * from './clustering/types';
export { KMeans } from './clustering/kmeans';
export { SpectralClustering } from './clustering/spectral';
export { AgglomerativeClustering } from './clustering/agglomerative';
export { pairwiseDistanceMatrix } from './utils/pairwise_distance';
export { findOptimalClusters } from './utils/findOptimalClusters';

/**
 * Main clustering namespace
 */
export const Clustering = {
  /**
   * Initialize the clustering library with the specified backend
   * 
   * @param config - Backend configuration options
   * @returns Promise that resolves when the backend is ready
   * 
   * @example
   * ```typescript
   * // Auto-detect best backend
   * await Clustering.init();
   * 
   * // Use specific backend
   * await Clustering.init({ backend: 'webgl' });
   * 
   * // With custom flags
   * await Clustering.init({
   *   backend: 'wasm',
   *   flags: { 'WASM_HAS_SIMD_SUPPORT': true }
   * });
   * ```
   */
  async init(config: BackendConfig = {}): Promise<void> {
    await initializeBackend(config);
  },
  
  // Re-export algorithms as properties for convenient access
  KMeans: KMeans,
  SpectralClustering: SpectralClustering,
  AgglomerativeClustering: AgglomerativeClustering,
};