import type * as tf_type from '@tensorflow/tfjs-core';
import type { TensorFlowBackend, Platform, ReactNativeConfig } from './platform_types';
import { is_react_native, is_node } from './platform';

let tf_instance: typeof tf_type | null = null;
let initialization_promise: Promise<typeof tf_type> | null = null;

export interface BackendConfig {
  backend?: TensorFlowBackend;
  flags?: Record<string, unknown>;
  react_native?: ReactNativeConfig;
  force_platform?: Platform;
}

export async function initialize_backend(config: BackendConfig = {}): Promise<typeof tf_type> {
  if (tf_instance) {
    return tf_instance;
  }

  if (initialization_promise) {
    return initialization_promise;
  }

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

export function ensure_backend(): typeof tf_type {
  if (tf_instance) return tf_instance;

  // If async init is in progress, don't race it with a sync load
  if (initialization_promise) {
    throw new Error(
      'TensorFlow.js is being initialized asynchronously via Clustering.init(). ' +
      'Await the init() call before using clustering algorithms.'
    );
  }

  if (is_node()) {
    tf_instance = load_backend_sync();
    return tf_instance;
  }

  // Browser/RN: accept a TF.js instance already loaded via <script> tag.
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

export function is_initialized(): boolean {
  return tf_instance !== null;
}

export function reset_backend(): void {
  tf_instance = null;
  initialization_promise = null;
}

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

async function load_backend(config: BackendConfig): Promise<typeof tf_type> {
  const platform_is_react_native = config.force_platform === 'react-native' || is_react_native();
  const platform_is_node = config.force_platform === 'node' || (!platform_is_react_native && is_node());

  // Browser is the only platform that code-splits per backend and must
  // feature-detect and verify WebGPU; node and react-native load a monolithic
  // package, so they share the generic flag + setBackend + ready tail below
  // while the browser loader owns that sequence for its own path.
  if (!platform_is_react_native && !platform_is_node) {
    const loader = await import('./loader.browser');
    return loader.load_tensor_flow(config.backend, config.flags);
  }

  let tf: typeof tf_type;
  if (platform_is_react_native) {
    const loader = await import(/* webpackIgnore: true */ './loader.rn');
    tf = await loader.load_tensor_flow();
  } else {
    const loader = await import(/* webpackIgnore: true */ './loader.node');
    tf = await loader.load_tensor_flow();
  }

  if (config.flags) {
    Object.entries(config.flags).forEach(([flag, value]) => {
      tf.env().setFlags({ [flag]: value as string | number | boolean });
    });
  }

  if (config.backend) {
    await tf.setBackend(config.backend);
  }

  await tf.ready();

  return tf;
}
