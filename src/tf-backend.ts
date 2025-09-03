/**
 * TensorFlow.js backend manager
 * 
 * Manages a singleton instance of TensorFlow.js with support for
 * multiple backends and environments.
 */

import type * as tfType from '@tensorflow/tfjs-core';
import type { TensorFlowBackend, Platform, ReactNativeConfig } from './types/platform';

// Singleton storage
let tfInstance: typeof tfType | null = null;
let initializationPromise: Promise<typeof tfType> | null = null;

/**
 * Backend configuration options
 */
export interface BackendConfig {
  /**
   * Preferred backend to use. If not specified, will auto-detect.
   * Options: 'cpu', 'webgl', 'wasm', 'node', 'node-gpu', 'rn-webgl'
   */
  backend?: TensorFlowBackend;
  
  /**
   * Custom flags to pass to the backend
   */
  flags?: Record<string, unknown>;
  
  /**
   * React Native specific configuration
   */
  reactNative?: ReactNativeConfig;
  
  /**
   * Force a specific platform detection (for testing)
   */
  forcePlatform?: Platform;
}

/**
 * Initialize the TensorFlow.js backend
 */
export async function initializeBackend(config: BackendConfig = {}): Promise<typeof tfType> {
  // Return existing instance if already initialized
  if (tfInstance) {
    return tfInstance;
  }
  
  // Return existing initialization promise if in progress
  if (initializationPromise) {
    return initializationPromise;
  }
  
  // Start initialization
  initializationPromise = loadBackend(config);
  
  try {
    tfInstance = await initializationPromise;
    return tfInstance;
  } catch (error) {
    // Reset on error to allow retry
    initializationPromise = null;
    throw error;
  }
}

/**
 * Get the current TensorFlow instance
 * @throws Error if not initialized
 */
export function getTensorFlow(): typeof tfType {
  if (!tfInstance) {
    throw new Error(
      'TensorFlow.js not initialized. Please call Clustering.init() first.'
    );
  }
  return tfInstance;
}

/**
 * Check if TensorFlow is initialized
 */
export function isInitialized(): boolean {
  return tfInstance !== null;
}

/**
 * Reset the backend (mainly for testing)
 */
export function resetBackend(): void {
  tfInstance = null;
  initializationPromise = null;
}

/**
 * Load the appropriate backend based on environment and config
 */
async function loadBackend(config: BackendConfig): Promise<typeof tfType> {
  // Detect environment
  const isReactNative = typeof navigator !== 'undefined' && 
                        navigator.product === 'ReactNative';
  const isNode = !isReactNative &&
                 typeof window === 'undefined' && 
                 typeof process !== 'undefined' && 
                 process.versions && 
                 process.versions.node;
  
  let tf: typeof tfType;
  
  if (isReactNative) {
    // React Native environment
    const loader = await import('./tf-loader.rn');
    tf = await loader.loadTensorFlow();
  } else if (isNode) {
    // Node.js environment
    const loader = await import('./tf-loader.node');
    tf = await loader.loadTensorFlow();
  } else {
    // Browser environment
    const loader = await import('./tf-loader.browser');
    tf = await loader.loadTensorFlow();
  }
  
  // Set custom flags if provided
  if (config.flags) {
    Object.entries(config.flags).forEach(([flag, value]) => {
      tf.env().setFlags({ [flag]: value as string | number | boolean });
    });
  }
  
  // Set specific backend if requested
  if (config.backend) {
    await tf.setBackend(config.backend);
  }
  
  // Wait for backend to be ready
  await tf.ready();
  
  return tf;
}