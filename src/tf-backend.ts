/**
 * TensorFlow.js backend manager
 *
 * Manages a singleton instance of TensorFlow.js with support for
 * multiple backends and environments.
 *
 * All TF.js access goes through this module:
 * - Explicit: await Clustering.init({ backend: 'wasm' }) before using algorithms
 * - Implicit: ensureBackend() auto-loads the best available backend on first use
 */

import type * as tfType from '@tensorflow/tfjs-core';
import type { TensorFlowBackend, Platform, ReactNativeConfig } from './types/platform';
import { isReactNative, isNode } from './utils/platform';

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
 * Initialize the TensorFlow.js backend (async, explicit path).
 * Call via Clustering.init() before using algorithms to control which backend is used.
 */
export async function initializeBackend(config: BackendConfig = {}): Promise<typeof tfType> {
  // If already initialized and user wants a specific backend, switch to it
  if (tfInstance && config.backend) {
    await tfInstance.setBackend(config.backend);
    await tfInstance.ready();
    return tfInstance;
  }

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
 * Get the current TensorFlow instance, or auto-load one synchronously.
 * This is the primary entry point used by tf-adapter.ts wrappers.
 *
 * In Node.js: synchronously loads the best available backend via require().
 * In browser/RN: throws if Clustering.init() has not been called.
 */
export function ensureBackend(): typeof tfType {
  if (tfInstance) return tfInstance;

  // If async init is in progress, don't race it with a sync load
  if (initializationPromise) {
    throw new Error(
      'TensorFlow.js is being initialized asynchronously via Clustering.init(). ' +
      'Await the init() call before using clustering algorithms.'
    );
  }

  // Auto-load synchronously in Node.js
  if (isNode()) {
    tfInstance = loadBackendSync();
    return tfInstance;
  }

  // In browser/RN, check if a TF.js backend was loaded via <script> tag
  // by probing for a globally available tf object with a registered backend.
  const g = globalThis as Record<string, unknown>;
  if (g['tf'] && typeof (g['tf'] as Record<string, unknown>)['getBackend'] === 'function') {
    const globalTf = g['tf'] as typeof tfType;
    if (globalTf.getBackend()) {
      tfInstance = globalTf;
      return tfInstance;
    }
  }

  throw new Error(
    'TensorFlow.js backend not initialized. Call await Clustering.init() first, ' +
    'or load TensorFlow.js via a <script> tag before using clustering algorithms.'
  );
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
 * Synchronous backend loading for Node.js (fallback when init() not called).
 * Uses require() with the same fallback chain as tf-loader.node.ts.
 */
function loadBackendSync(): typeof tfType {
  try {
    require.resolve('@tensorflow/tfjs-node');
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    return require('@tensorflow/tfjs-node') as typeof tfType;
  } catch {
    try {
      require.resolve('@tensorflow/tfjs');
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      return require('@tensorflow/tfjs') as typeof tfType;
    } catch {
      try {
        require.resolve('@tensorflow/tfjs-core');
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        return require('@tensorflow/tfjs-core') as typeof tfType;
      } catch {
        throw new Error(
          'No TensorFlow.js backend available. Install one of:\n' +
          '- @tensorflow/tfjs-node (for CPU acceleration)\n' +
          '- @tensorflow/tfjs (for pure JavaScript fallback)\n' +
          'Or call Clustering.init() after installing a backend.'
        );
      }
    }
  }
}

/**
 * Load the appropriate backend based on environment and config (async path)
 */
async function loadBackend(config: BackendConfig): Promise<typeof tfType> {
  // Use platform utilities for consistent detection
  const platformIsReactNative = config.forcePlatform === 'react-native' || isReactNative();
  const platformIsNode = config.forcePlatform === 'node' || (!platformIsReactNative && isNode());

  let tf: typeof tfType;

  if (platformIsReactNative) {
    // React Native environment
    const loader = await import(/* webpackIgnore: true */ './tf-loader.rn');
    tf = await loader.loadTensorFlow();
  } else if (platformIsNode) {
    // Node.js environment
    const loader = await import(/* webpackIgnore: true */ './tf-loader.node');
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