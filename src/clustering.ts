/**
 * Main entry point for the clustering library
 * 
 * Provides initialization and configuration for multi-platform support.
 */

import { initializeBackend, resetBackend, BackendConfig } from './tf-backend';
import { KMeans } from './clustering/kmeans';
import { SpectralClustering } from './clustering/spectral';
import { AgglomerativeClustering } from './clustering/agglomerative';
import { SOM } from './clustering/som';
import type { Platform, PlatformFeatures } from './types/platform';
import { getPlatform } from './utils/platform';

// ---------------------------------------------------------------------------
// Idempotency guard state for Clustering.init()
// ---------------------------------------------------------------------------
let initPromise: Promise<void> | null = null;
let initConfigKey: string | null = null;

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

/**
 * Deterministic JSON serialization with sorted keys at every nesting level.
 * Used to produce a stable string for config comparison.
 */
function sortedStringify(value: unknown): string {
  if (value === null || value === undefined) return String(value);
  if (typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return '[' + value.map(sortedStringify).join(',') + ']';
  if (!isPlainObject(value)) return JSON.stringify(value);
  const keys = Object.keys(value).sort();
  const pairs = keys
    .filter(k => value[k] !== undefined)
    .map(k => `${JSON.stringify(k)}:${sortedStringify(value[k])}`);
  return `{${pairs.join(',')}}`;
}

/** Return the number of defined (non-undefined) values in an object. */
function definedKeyCount(obj: object): number {
  return Object.entries(obj).filter(([, v]) => v !== undefined).length;
}

/**
 * Strip empty sub-objects so that `{}`, `{ flags: undefined }`,
 * `{ flags: {} }`, and `{ flags: { X: undefined } }` all normalize
 * identically.
 */
function stripEmpty(config: BackendConfig): BackendConfig {
  const result: BackendConfig = {};
  if (config.backend !== undefined) result.backend = config.backend;
  if (config.forcePlatform !== undefined) result.forcePlatform = config.forcePlatform;
  if (config.flags !== undefined && definedKeyCount(config.flags) > 0) {
    result.flags = config.flags;
  }
  if (config.reactNative !== undefined && definedKeyCount(config.reactNative) > 0) {
    result.reactNative = config.reactNative;
  }
  return result;
}

/** Produce a stable config key for comparison. */
function configKey(config: BackendConfig): string {
  return sortedStringify(stripEmpty(config));
}

// Detect platform at runtime using utility function
const detectPlatform = (): Platform => {
  return getPlatform();
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
    case 'react-native':
      return {
        gpuAcceleration: true, // rn-webgl provides GPU acceleration
        wasmSimd: false,
        nodeBindings: false,
        webgl: false, // Uses rn-webgl instead
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
   * Current platform (detected at runtime)
   */
  platform: detectPlatform(),
  
  /**
   * Platform features
   */
  features: getPlatformFeatures(detectPlatform()),
  
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
    const key = configKey(config);

    if (initPromise) {
      if (initConfigKey !== key) {
        throw new Error(
          'Clustering.init() has already been called with a different configuration. ' +
          'Call Clustering.reset() before re-initializing with new options.'
        );
      }
      return initPromise;
    }

    initConfigKey = key;
    initPromise = initializeBackend(config).then(
      () => {},
      (err: unknown) => {
        // Reset on failure to allow retry
        initPromise = null;
        initConfigKey = null;
        throw err;
      },
    );

    return initPromise;
  },

  /**
   * Reset initialization state, allowing re-initialization with a different
   * config. After calling this, {@link Clustering.init | init()} must be
   * called again before using clustering algorithms.
   */
  reset(): void {
    resetBackend();
    initPromise = null;
    initConfigKey = null;
  },

  // Re-export algorithms as properties for convenient access
  KMeans: KMeans,
  SpectralClustering: SpectralClustering,
  AgglomerativeClustering: AgglomerativeClustering,
  SOM: SOM,
};