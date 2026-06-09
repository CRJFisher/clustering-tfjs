/**
 * Main entry point for the clustering library
 * 
 * Provides initialization and configuration for multi-platform support.
 */

import { initialize_backend, reset_backend, BackendConfig } from '../backend/backend';
import { KMeans } from './kmeans';
import { SpectralClustering } from './spectral';
import { AgglomerativeClustering } from './agglomerative';
import { HDBSCAN } from './hdbscan';
import { SOM } from './som';
import type { Platform, PlatformFeatures } from '../backend/platform_types';
import { get_platform } from '../backend/platform';

// ---------------------------------------------------------------------------
// Idempotency guard state for Clustering.init()
// ---------------------------------------------------------------------------
let init_promise: Promise<void> | null = null;
let init_config_key: string | null = null;

function is_plain_object(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

/**
 * Deterministic JSON serialization with sorted keys at every nesting level.
 * Used to produce a stable string for config comparison.
 */
function sorted_stringify(value: unknown): string {
  if (value === null || value === undefined) return String(value);
  if (typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return '[' + value.map(sorted_stringify).join(',') + ']';
  if (!is_plain_object(value)) return JSON.stringify(value);
  const keys = Object.keys(value).sort();
  const pairs = keys
    .filter(k => value[k] !== undefined)
    .map(k => `${JSON.stringify(k)}:${sorted_stringify(value[k])}`);
  return `{${pairs.join(',')}}`;
}

/** Return the number of defined (non-undefined) values in an object. */
function defined_key_count(obj: object): number {
  return Object.entries(obj).filter(([, v]) => v !== undefined).length;
}

/**
 * Strip empty sub-objects so that `{}`, `{ flags: undefined }`,
 * `{ flags: {} }`, and `{ flags: { X: undefined } }` all normalize
 * identically.
 */
function strip_empty(config: BackendConfig): BackendConfig {
  const result: BackendConfig = {};
  if (config.backend !== undefined) result.backend = config.backend;
  if (config.force_platform !== undefined) result.force_platform = config.force_platform;
  if (config.flags !== undefined && defined_key_count(config.flags) > 0) {
    result.flags = config.flags;
  }
  if (config.react_native !== undefined && defined_key_count(config.react_native) > 0) {
    result.react_native = config.react_native;
  }
  return result;
}

/** Produce a stable config key for comparison. */
function config_key(config: BackendConfig): string {
  return sorted_stringify(strip_empty(config));
}

// Detect platform at runtime using utility function
const detect_platform = (): Platform => {
  return get_platform();
};

// Get platform features based on detected platform
const get_platform_features = (platform: Platform): PlatformFeatures => {
  switch (platform) {
    case 'browser':
      return {
        gpu_acceleration: typeof WebGLRenderingContext !== 'undefined',
        wasm_simd: typeof WebAssembly !== 'undefined' && 'validate' in WebAssembly,
        node_bindings: false,
        webgl: typeof WebGLRenderingContext !== 'undefined',
      };
    case 'node':
      return {
        gpu_acceleration: false, // Will be updated after backend init
        wasm_simd: false,
        node_bindings: true,
        webgl: false,
      };
    case 'react-native':
      return {
        gpu_acceleration: true, // rn-webgl provides GPU acceleration
        wasm_simd: false,
        node_bindings: false,
        webgl: false, // Uses rn-webgl instead
      };
    default:
      return {
        gpu_acceleration: false,
        wasm_simd: false,
        node_bindings: false,
        webgl: false,
      };
  }
};

/**
 * Main clustering namespace with platform awareness
 */
export const Clustering = {
  /**
   * Current platform (detected at runtime)
   */
  platform: detect_platform(),
  
  /**
   * Platform features
   */
  features: get_platform_features(detect_platform()),
  
  /**
   * Initialize the clustering library with the specified backend.
   *
   * **Idempotent**: concurrent calls with the same (or equivalent) config
   * return the exact same `Promise` object. A second call after initialization
   * completes with the same config is a no-op that resolves immediately.
   *
   * Calling with a *different* config while initialization is in progress or
   * after it has completed throws a synchronous error. Call
   * {@link Clustering.reset | reset()} first to re-initialize with a new
   * configuration.
   *
   * @param config - Backend configuration options
   * @returns Promise that resolves when the backend is ready
   * @throws {Error} If called with a different config than a previous call
   *
   * @example
   * ```typescript
   * // Auto-detect best backend
   * await Clustering.init();
   *
   * // Use specific backend
   * await Clustering.init({ backend: 'webgl' });
   *
   * // Safe to call concurrently — both await the same initialization
   * await Promise.all([Clustering.init(), Clustering.init()]);
   *
   * // Different config after init throws — reset first
   * await Clustering.init({ backend: 'cpu' });
   * Clustering.reset();
   * await Clustering.init({ backend: 'wasm' });
   * ```
   */
  init(config: BackendConfig = {}): Promise<void> {
    const key = config_key(config);

    if (init_promise) {
      if (init_config_key !== key) {
        throw new Error(
          'Clustering.init() has already been called with a different configuration. ' +
          'Call Clustering.reset() before re-initializing with new options.'
        );
      }
      return init_promise;
    }

    init_config_key = key;
    init_promise = initialize_backend(config).then(
      () => {},
      (err: unknown) => {
        // Reset on failure to allow retry
        init_promise = null;
        init_config_key = null;
        throw err;
      },
    );

    return init_promise;
  },

  /**
   * Reset initialization state, allowing re-initialization with a different
   * config. After calling this, {@link Clustering.init | init()} must be
   * called again before using clustering algorithms.
   */
  reset(): void {
    reset_backend();
    init_promise = null;
    init_config_key = null;
  },

  // Re-export algorithms as properties for convenient access
  KMeans: KMeans,
  SpectralClustering: SpectralClustering,
  AgglomerativeClustering: AgglomerativeClustering,
  HDBSCAN: HDBSCAN,
  SOM: SOM,
};