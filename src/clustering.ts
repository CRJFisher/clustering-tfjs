/**
 * Main entry point for the clustering library
 * 
 * Provides initialization and configuration for multi-platform support.
 */

import { initializeBackend, BackendConfig } from './tf-backend';
import { KMeans } from './clustering/kmeans';
import { SpectralClustering } from './clustering/spectral';
import { AgglomerativeClustering } from './clustering/agglomerative';
import { SOM } from './clustering/som';
import type { Platform, DetectedPlatform, PlatformFeatures } from './clustering-types';

// Re-export all clustering algorithms and utilities
export * from './clustering/types';
export { KMeans } from './clustering/kmeans';
export { SpectralClustering } from './clustering/spectral';
export { AgglomerativeClustering } from './clustering/agglomerative';
export { SOM } from './clustering/som';
export { pairwiseDistanceMatrix } from './utils/pairwise_distance';
export { findOptimalClusters } from './utils/findOptimalClusters';

// Re-export advanced types
export type { Platform, DetectedPlatform, PlatformFeatures, ExtendedBackendConfig } from './clustering-types';

// Detect platform at runtime
const detectPlatform = (): Platform => {
  if (typeof window !== 'undefined' && typeof window.document !== 'undefined') {
    return 'browser';
  } else if (typeof process !== 'undefined' && process.versions && process.versions.node) {
    return 'node';
  }
  return 'unknown';
};

// Get platform features based on detected platform
const getPlatformFeatures = (platform: Platform): PlatformFeatures => {
  switch (platform) {
    case 'browser':
      return {
        gpuAcceleration: typeof WebGLRenderingContext !== 'undefined',
        wasmSimd: typeof WebAssembly !== 'undefined' && 'validate' in WebAssembly,
        nodeBindings: false,
        webgl: typeof WebGLRenderingContext !== 'undefined',
      };
    case 'node':
      return {
        gpuAcceleration: false, // Will be updated after backend init
        wasmSimd: false,
        nodeBindings: true,
        webgl: false,
      };
    default:
      return {
        gpuAcceleration: false,
        wasmSimd: false,
        nodeBindings: false,
        webgl: false,
      };
  }
};

/**
 * Main clustering namespace with platform awareness
 */
export const Clustering = {
  /**
   * Current platform
   */
  platform: detectPlatform() as DetectedPlatform,
  
  /**
   * Platform features
   */
  features: getPlatformFeatures(detectPlatform()),
  
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
    
    // Features are set at detection time
    // Could be enhanced later to detect actual backend capabilities
  },
  
  // Re-export algorithms as properties for convenient access
  KMeans: KMeans,
  SpectralClustering: SpectralClustering,
  AgglomerativeClustering: AgglomerativeClustering,
  SOM: SOM,
};