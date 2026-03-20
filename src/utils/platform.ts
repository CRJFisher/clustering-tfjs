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

/** In-memory storage for Node.js and React Native fallback */
const memoryStorage = new Map<string, string>();

/**
 * Platform-safe data persistence
 *
 * In browser, uses localStorage.
 * In Node.js and React Native, uses in-memory Map.
 */
export class PlatformStorage {
  private platform: Platform;

  constructor() {
    this.platform = getPlatform();
  }

  async setItem(key: string, value: string): Promise<void> {
    if (this.platform === 'browser') {
      localStorage.setItem(key, value);
    } else {
      memoryStorage.set(key, value);
    }
  }

  async getItem(key: string): Promise<string | null> {
    if (this.platform === 'browser') {
      return localStorage.getItem(key);
    }
    return memoryStorage.get(key) ?? null;
  }

  async removeItem(key: string): Promise<void> {
    if (this.platform === 'browser') {
      localStorage.removeItem(key);
    } else {
      memoryStorage.delete(key);
    }
  }
}

/**
 * Platform-safe data fetching
 */
export async function platformFetch(url: string, options?: RequestInit): Promise<Response> {
  if (typeof fetch !== 'undefined') {
    return fetch(url, options);
  }
  throw new Error('Fetch not available in this environment. Node.js 18+ and modern browsers provide global fetch.');
}
