import type { Platform } from './platform_types';

export function is_node(): boolean {
  return typeof process !== 'undefined' &&
         process.versions !== undefined &&
         process.versions.node !== undefined &&
         typeof window === 'undefined';
}

// Multiple globals are checked because different RN JS engines (Hermes, JSC, V8)
// expose different signals; no single one is universally reliable.
export function is_react_native(): boolean {
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

export function get_platform(): Platform {
  if (is_react_native()) return 'react-native';
  if (is_node()) return 'node';
  return 'browser';
}
