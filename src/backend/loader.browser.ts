import type * as tf_core from '@tensorflow/tfjs-core';

// Each browser backend is reached through its own static-specifier dynamic
// import so a code-splitting bundler emits one chunk per backend — requesting a
// single backend never pulls the others into the bundle. `@tensorflow/tfjs-core`
// supplies the engine and ops; importing a backend package registers that
// backend with the shared core engine as an import side-effect.
const BROWSER_BACKEND_IMPORTERS: Record<string, () => Promise<unknown>> = {
  cpu: () => import('@tensorflow/tfjs-backend-cpu'),
  webgl: () => import('@tensorflow/tfjs-backend-webgl'),
  webgpu: () => import('@tensorflow/tfjs-backend-webgpu'),
};

const DEFAULT_BROWSER_BACKEND = 'webgl';

export async function load_tensor_flow(
  backend?: string,
  flags?: Record<string, unknown>,
): Promise<typeof tf_core> {
  if (typeof window !== 'undefined') {
    const global_window: Window & { tf?: typeof tf_core } = window;
    if (global_window.tf) {
      return global_window.tf;
    }
  }

  const requested = backend ?? DEFAULT_BROWSER_BACKEND;
  const importer = BROWSER_BACKEND_IMPORTERS[requested];
  if (!importer) {
    throw new Error(
      `Unsupported browser backend '${requested}'. ` +
      `Supported browser backends: ${Object.keys(BROWSER_BACKEND_IMPORTERS).join(', ')}.`,
    );
  }

  // Feature-detect before importing the heavy package so a missing-WebGPU
  // environment fails cleanly and the caller can fall back to webgl.
  if (requested === 'webgpu' && !has_webgpu()) {
    throw new Error(
      'WebGPU is not available in this environment (navigator.gpu is undefined). ' +
      "Initialize with { backend: 'webgl' } or { backend: 'cpu' } instead.",
    );
  }

  const tf = await load_core();

  if (flags) {
    Object.entries(flags).forEach(([flag, value]) => {
      tf.env().setFlags({ [flag]: value as string | number | boolean });
    });
  }

  await importer();

  // setBackend resolves to false when the backend registered but failed to
  // initialize and tf silently kept a previously-registered backend; the
  // getBackend re-check is the authoritative guard that the lane is real.
  const set_ok = await tf.setBackend(requested);
  await tf.ready();
  if (!set_ok || tf.getBackend() !== requested) {
    throw new Error(
      `Failed to initialize the '${requested}' TensorFlow.js backend ` +
      `(setBackend returned ${set_ok}, active backend is '${tf.getBackend()}').`,
    );
  }

  return tf;
}

async function load_core(): Promise<typeof tf_core> {
  try {
    return await import('@tensorflow/tfjs-core');
  } catch {
    throw new Error(
      'TensorFlow.js not found. Install @tensorflow/tfjs-core and a backend package:\n' +
      'npm install @tensorflow/tfjs-core @tensorflow/tfjs-backend-webgl'
    );
  }
}

function has_webgpu(): boolean {
  if (typeof navigator === 'undefined') return false;
  const nav: Navigator & { gpu?: unknown } = navigator;
  return nav.gpu != null;
}
