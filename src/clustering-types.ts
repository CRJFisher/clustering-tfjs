/**
 * Advanced TypeScript type definitions for multi-platform support
 * 
 * This file provides conditional types and module augmentation for
 * platform-aware type safety across browser and Node.js environments.
 */

import type * as tf from '@tensorflow/tfjs-core';

/**
 * Platform detection type
 */
export type Platform = 'browser' | 'node' | 'unknown';

/**
 * Detect platform at compile time based on global objects
 */
export type DetectedPlatform = 
  typeof globalThis extends { window: unknown } ? 'browser' :
  typeof globalThis extends { process: unknown } ? 'node' :
  'unknown';

/**
 * Backend-specific features available in different environments
 */
export interface BackendFeatures {
  gpuAcceleration: boolean;
  wasmSimd: boolean;
  nodeBindings: boolean;
  webgl: boolean;
}

/**
 * Conditional backend features based on platform
 */
export type PlatformFeatures<_P extends Platform = DetectedPlatform> = BackendFeatures;

/**
 * Extended backend configuration with platform awareness
 */
export interface ExtendedBackendConfig<P extends Platform = DetectedPlatform> {
  backend?: P extends 'browser' ? 'cpu' | 'webgl' | 'wasm' :
            P extends 'node' ? 'cpu' | 'tensorflow' :
            string;
  flags?: Record<string, unknown>;
  platform?: P;
}

/**
 * Type-safe clustering namespace with platform awareness
 */
export interface ClusteringNamespace<P extends Platform = DetectedPlatform> {
  init(config?: ExtendedBackendConfig<P>): Promise<void>;
  KMeans: typeof import('./clustering/kmeans').KMeans;
  SpectralClustering: typeof import('./clustering/spectral').SpectralClustering;
  AgglomerativeClustering: typeof import('./clustering/agglomerative').AgglomerativeClustering;
  
  // Platform-specific properties
  platform: P;
  features: PlatformFeatures<P>;
}

/**
 * Module augmentation for Node.js specific features
 */
declare module '@tensorflow/tfjs-core' {
  interface TensorFlow {
    // Node.js specific properties
    nodeBackend?: {
      isGPU: boolean;
      binding: unknown;
    };
  }
}

/**
 * Utility type for tensor operations that may vary by platform
 */
export type PlatformTensor<_T, P extends Platform = DetectedPlatform> = 
  P extends 'browser' ? tf.Tensor<tf.Rank> :
  P extends 'node' ? tf.Tensor<tf.Rank> & { _nodeData?: Buffer } :
  tf.Tensor<tf.Rank>;

/**
 * Type guard for platform detection
 */
export function isPlatform<P extends Platform>(
  platform: Platform,
  target: P
): platform is P {
  return platform === target;
}

/**
 * Type-safe backend initialization result
 */
export interface InitializationResult<P extends Platform = DetectedPlatform> {
  platform: P;
  backend: string;
  features: PlatformFeatures<P>;
  tf: typeof tf;
}