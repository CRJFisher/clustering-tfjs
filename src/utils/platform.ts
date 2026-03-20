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
