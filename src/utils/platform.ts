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
 * Safely check if running in React Native environment
 */
export function isReactNative(): boolean {
  return typeof navigator !== 'undefined' && 
         navigator.product === 'ReactNative';
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
      // In React Native, AsyncStorage would be used
      // For now, we'll use in-memory storage as fallback
      if (typeof (global as any).AsyncStorage !== 'undefined') {
        await (global as any).AsyncStorage.setItem(key, value);
      } else {
        // Fallback to in-memory storage
        (global as any).__platformStorage = (global as any).__platformStorage || {};
        (global as any).__platformStorage[key] = value;
      }
    } else if (this.platform === 'browser') {
      localStorage.setItem(key, value);
    } else {
      // Node.js - use in-memory storage for now
      (global as any).__platformStorage = (global as any).__platformStorage || {};
      (global as any).__platformStorage[key] = value;
    }
  }
  
  async getItem(key: string): Promise<string | null> {
    if (this.platform === 'react-native') {
      if (typeof (global as any).AsyncStorage !== 'undefined') {
        return await (global as any).AsyncStorage.getItem(key);
      } else {
        const storage = (global as any).__platformStorage || {};
        return storage[key] || null;
      }
    } else if (this.platform === 'browser') {
      return localStorage.getItem(key);
    } else {
      const storage = (global as any).__platformStorage || {};
      return storage[key] || null;
    }
  }
  
  async removeItem(key: string): Promise<void> {
    if (this.platform === 'react-native') {
      if (typeof (global as any).AsyncStorage !== 'undefined') {
        await (global as any).AsyncStorage.removeItem(key);
      } else {
        const storage = (global as any).__platformStorage || {};
        delete storage[key];
      }
    } else if (this.platform === 'browser') {
      localStorage.removeItem(key);
    } else {
      const storage = (global as any).__platformStorage || {};
      delete storage[key];
    }
  }
}

/**
 * Platform-safe data fetching
 */
export async function platformFetch(url: string, options?: RequestInit): Promise<Response> {
  if (typeof fetch !== 'undefined') {
    // Modern browsers and React Native have fetch
    return fetch(url, options);
  } else if (isNode()) {
    // Node.js might need a polyfill
    try {
      const nodeFetch = await import('node-fetch');
      return nodeFetch.default(url, options) as Promise<Response>;
    } catch {
      throw new Error('Fetch not available. Install node-fetch for Node.js environments.');
    }
  } else {
    throw new Error('Fetch not available in this environment');
  }
}