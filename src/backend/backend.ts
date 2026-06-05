/**
 * TensorFlow.js backend manager
 *
 * Manages a singleton instance of TensorFlow.js with support for
 * multiple backends and environments.
 *
 * All TF.js access goes through this module:
 * - Explicit: await Clustering.init({ backend: 'wasm' }) before using algorithms
 * - Implicit: ensure_backend() auto-loads the best available backend on first use
 */

import type * as tf_type from '@tensorflow/tfjs-core';
import type { TensorFlowBackend, Platform, ReactNativeConfig } from './platform_types';
import { is_react_native, is_node } from './platform';

// Singleton storage
let tf_instance: typeof tf_type | null = null;
let initialization_promise: Promise<typeof tf_type> | null = null;

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
  react_native?: ReactNativeConfig;

  /**
   * Force a specific platform detection (for testing)
   */
  force_platform?: Platform;
}

/**
 * Initialize the TensorFlow.js backend (async, explicit path).
 * Typically called indirectly via {@link Clustering.init}.
 *
 * Idempotent: returns the cached instance if already initialized, or the
 * in-flight promise if initialization is in progress. Config comparison and
 * conflict detection are handled by `Clustering.init()`.
 */
export async function initialize_backend(config: BackendConfig = {}): Promise<typeof tf_type> {
  // Return existing instance if already initialized
  if (tf_instance) {
    return tf_instance;
  }

  // Return existing initialization promise if in progress
  if (initialization_promise) {
    return initialization_promise;
  }

  // Start initialization
  initialization_promise = load_backend(config);

  try {
    tf_instance = await initialization_promise;
    return tf_instance;
  } catch (error) {
    // Reset on error to allow retry
    initialization_promise = null;
    throw error;
  }
}

/**
 * Get the current TensorFlow instance, or auto-load one synchronously.
 * This is the primary entry point used by adapter.ts wrappers.
 *
 * In Node.js: synchronously loads the best available backend via require().
 * In browser/RN: throws if Clustering.init() has not been called.
 */
export function ensure_backend(): typeof tf_type {
  if (tf_instance) return tf_instance;

  // If async init is in progress, don't race it with a sync load
  if (initialization_promise) {
    throw new Error(
      'TensorFlow.js is being initialized asynchronously via Clustering.init(). ' +
      'Await the init() call before using clustering algorithms.'
    );
  }

  // Auto-load synchronously in Node.js
  if (is_node()) {
    tf_instance = load_backend_sync();
    return tf_instance;
  }

  // In browser/RN, check if a TF.js backend was loaded via <script> tag
  // by probing for a globally available tf object with a registered backend.
  const g = globalThis as Record<string, unknown>;
  if (g['tf'] && typeof (g['tf'] as Record<string, unknown>)['getBackend'] === 'function') {
    const global_tf = g['tf'] as typeof tf_type;
    if (global_tf.getBackend()) {
      tf_instance = global_tf;
      return tf_instance;
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
export function is_initialized(): boolean {
  return tf_instance !== null;
}

/**
 * Reset the backend (mainly for testing)
 */
export function reset_backend(): void {
  tf_instance = null;
  initialization_promise = null;
}

/**
 * Synchronous backend loading for Node.js (fallback when init() not called).
 * Uses require() with the same fallback chain as loader.node.ts.
 */
function load_backend_sync(): typeof tf_type {
  try {
    require.resolve('@tensorflow/tfjs-node');
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    return require('@tensorflow/tfjs-node') as typeof tf_type;
  } catch {
    try {
      require.resolve('@tensorflow/tfjs');
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      return require('@tensorflow/tfjs') as typeof tf_type;
    } catch {
      try {
        require.resolve('@tensorflow/tfjs-core');
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        return require('@tensorflow/tfjs-core') as typeof tf_type;
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
async function load_backend(config: BackendConfig): Promise<typeof tf_type> {
  // Use platform utilities for consistent detection
  const platform_is_react_native = config.force_platform === 'react-native' || is_react_native();
  const platform_is_node = config.force_platform === 'node' || (!platform_is_react_native && is_node());

  let tf: typeof tf_type;

  if (platform_is_react_native) {
    // React Native environment
    const loader = await import(/* webpackIgnore: true */ './loader.rn');
    tf = await loader.load_tensor_flow();
  } else if (platform_is_node) {
    // Node.js environment
    const loader = await import(/* webpackIgnore: true */ './loader.node');
    tf = await loader.load_tensor_flow();
  } else {
    // Browser environment
    const loader = await import('./loader.browser');
    tf = await loader.load_tensor_flow();
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