import type * as tf_core from '@tensorflow/tfjs-core';
import type { TensorFlowBackend } from './platform_types';

// Each browser backend is reached through its own static-specifier dynamic
// import so a code-splitting bundler can emit a separate chunk per backend and
// a caller requesting one backend never network-loads the webgl OR webgpu lane
// it did not ask for. (cpu is a dependency of both webgl and webgpu, so its
// kernels ride along as the fallback whenever either GPU lane is selected.)
// `@tensorflow/tfjs-core` supplies the engine and ops; importing a backend
// package registers that backend with the shared core engine as a side-effect.
const BROWSER_BACKEND_IMPORTERS: Partial<Record<TensorFlowBackend, () => Promise<unknown>>> = {
  cpu: () => import('@tensorflow/tfjs-backend-cpu'),
  webgl: () => import('@tensorflow/tfjs-backend-webgl'),
  webgpu: () => import('@tensorflow/tfjs-backend-webgpu'),
  wasm: () => import('@tensorflow/tfjs-backend-wasm'),
};

// webgl is the broadest GPU-accelerated browser backend and matches the
// behaviour of the monolithic build this loader replaces; webgpu is opt-in
// behind feature detection and cpu is the slow universal fallback.
const DEFAULT_BROWSER_BACKEND: TensorFlowBackend = 'webgl';

export async function load_tensor_flow(
  backend?: TensorFlowBackend,
  flags?: Record<string, unknown>,
): Promise<typeof tf_core> {
  // A user who loaded tfjs via a <script> tag exposes an initialized window.tf;
  // respect it rather than loading our own, unless they explicitly asked for a
  // backend it is not running (then fall through and load that backend).
  if (typeof window !== 'undefined') {
    const global_window: Window & { tf?: typeof tf_core } = window;
    const global_tf = global_window.tf;
    if (global_tf && global_tf.getBackend() && (backend === undefined || global_tf.getBackend() === backend)) {
      return global_tf;
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
