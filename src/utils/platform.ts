/**
 * Platform-safe utility functions
 * 
 * Provides platform-agnostic utilities that work across
 * browser, Node.js, and React Native environments.
 */

import type { Platform } from '../types/platform';

/**
 * Safely check if running in Node.js environment
 */
export function isNode(): boolean {
  return typeof process !== 'undefined' &&
         process.versions !== undefined &&
         process.versions.node !== undefined &&
         typeof window === 'undefined';
}

/**
 * Safely check if running in React Native environment.
 * Uses multiple signals for robust detection across Hermes, JSC, and V8 engines.
 */
export function isReactNative(): boolean {
  if (typeof globalThis === 'undefined') return false;

  const g: Record<string, unknown> = globalThis as Record<string, unknown>;

  // HermesInternal: present on the Hermes engine (default since RN 0.70)
  if (typeof g['HermesInternal'] !== 'undefined') return true;

  // __fbBatchedBridge: the core React Native bridge, present in all RN environments
  if (typeof g['__fbBatchedBridge'] !== 'undefined') return true;

  // nativeCallSyncHook: synchronous native call mechanism in RN
  if (typeof g['nativeCallSyncHook'] !== 'undefined') return true;

  return false;
}

/**
 * Safely check if running in browser environment
 */
export function isBrowser(): boolean {
  return typeof window !== 'undefined' && 
         !isReactNative() &&
         !isNode();
}

/**
 * Get current platform
 */
export function getPlatform(): Platform {
  if (isReactNative()) return 'react-native';
  if (isNode()) return 'node';
  return 'browser';
}

/**
 * Safely check if running on Windows (Node.js only)
 */
export function isWindows(): boolean {
  return isNode() && process.platform === 'win32';
}

/**
 * Safely check if running in CI environment
 */
export function isCI(): boolean {
  if (!isNode()) return false;
  return process.env.CI === 'true' || 
         process.env.CI === '1' ||
         process.env.CONTINUOUS_INTEGRATION === 'true';
}

interface AsyncStorageLike {
  setItem(key: string, value: string): Promise<void>;
  getItem(key: string): Promise<string | null>;
  removeItem(key: string): Promise<void>;
}

interface PlatformGlobals {
  AsyncStorage?: AsyncStorageLike;
  __platformStorage?: Record<string, string>;
}

function getGlobals(): PlatformGlobals {
  return globalThis as unknown as PlatformGlobals;
}

/**
 * Platform-safe data persistence
 *
 * In React Native, this would use AsyncStorage
 * In Node.js, this could use file system
 * In browser, this uses localStorage
 */
export class PlatformStorage {
  private platform: Platform;

  constructor() {
    this.platform = getPlatform();
  }

  async setItem(key: string, value: string): Promise<void> {
    if (this.platform === 'react-native') {
      const g = getGlobals();
      if (g.AsyncStorage) {
        await g.AsyncStorage.setItem(key, value);
      } else {
        g.__platformStorage = g.__platformStorage || {};
        g.__platformStorage[key] = value;
      }
    } else if (this.platform === 'browser') {
      localStorage.setItem(key, value);
    } else {
      const g = getGlobals();
      g.__platformStorage = g.__platformStorage || {};
      g.__platformStorage[key] = value;
    }
  }

  async getItem(key: string): Promise<string | null> {
    if (this.platform === 'react-native') {
      const g = getGlobals();
      if (g.AsyncStorage) {
        return await g.AsyncStorage.getItem(key);
      } else {
        const storage = g.__platformStorage || {};
        return storage[key] || null;
      }
    } else if (this.platform === 'browser') {
      return localStorage.getItem(key);
    } else {
      const storage = getGlobals().__platformStorage || {};
      return storage[key] || null;
    }
  }

  async removeItem(key: string): Promise<void> {
    if (this.platform === 'react-native') {
      const g = getGlobals();
      if (g.AsyncStorage) {
        await g.AsyncStorage.removeItem(key);
      } else {
        const storage = g.__platformStorage || {};
        delete storage[key];
      }
    } else if (this.platform === 'browser') {
      localStorage.removeItem(key);
    } else {
      const storage = getGlobals().__platformStorage || {};
      delete storage[key];
    }
  }
}

/**
 * Platform-safe data fetching
 */
export async function platformFetch(url: string, options?: RequestInit): Promise<Response> {
  if (typeof fetch !== 'undefined') {
    // Modern browsers, React Native, and Node 18+ have fetch
    return fetch(url, options);
  } else if (isNode()) {
    // Fallback for older Node.js without native fetch
    try {
      const nodeFetch = await import('node-fetch');
      // node-fetch types diverge from DOM fetch types; cast through unknown
      const response = await nodeFetch.default(
        url,
        options as import('node-fetch').RequestInit
      );
      return response as unknown as Response;
    } catch {
      throw new Error('Fetch not available. Install node-fetch for Node.js environments.');
    }
  } else {
    throw new Error('Fetch not available in this environment');
  }
}