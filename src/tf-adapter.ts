/**
 * TensorFlow.js adapter module
 *
 * Provides a platform-agnostic, lazy-loaded interface to TensorFlow.js.
 * All function calls are deferred through ensureBackend() in tf-backend.ts,
 * which means:
 *
 * - Clustering.init({ backend: 'wasm' }) controls which backend is used
 * - Without explicit init(), auto-loads the best available backend on first use
 * - Works across Node.js, browser, and React Native without build-time swapping
 *
 * Only the TF.js functions actually used by clustering algorithms are exported.
 */

import type * as tfTypes from '@tensorflow/tfjs-core';
import { ensureBackend } from './tf-backend';

// ---------------------------------------------------------------------------
// Type re-exports (compile-time only, erased at runtime)
// ---------------------------------------------------------------------------
export type {
  Tensor,
  Tensor1D,
  Tensor2D,
  Tensor3D,
  Tensor4D,
  Tensor5D,
  Scalar,
  TensorLike,
  DataType,
  Rank,
  TensorBuffer,
} from '@tensorflow/tfjs-core';

// ---------------------------------------------------------------------------
// Lazy runtime wrappers — only functions used by clustering algorithms
// Each call goes through ensureBackend() to guarantee a backend is loaded.
// ---------------------------------------------------------------------------

// Tensor creation
export const tensor: typeof tfTypes.tensor = (...args) => ensureBackend().tensor(...args);
export const tensor1d: typeof tfTypes.tensor1d = (...args) => ensureBackend().tensor1d(...args);
export const tensor2d: typeof tfTypes.tensor2d = (...args) => ensureBackend().tensor2d(...args);
export const tensor3d: typeof tfTypes.tensor3d = (...args) => ensureBackend().tensor3d(...args);
export const scalar: typeof tfTypes.scalar = (...args) => ensureBackend().scalar(...args);
export const zeros: typeof tfTypes.zeros = (...args) => ensureBackend().zeros(...args);
export const ones: typeof tfTypes.ones = (...args) => ensureBackend().ones(...args);
export const onesLike: typeof tfTypes.onesLike = (...args) => ensureBackend().onesLike(...args);
export const fill: typeof tfTypes.fill = (...args) => ensureBackend().fill(...args);
export const eye: typeof tfTypes.eye = (...args) => ensureBackend().eye(...args);
export const linspace: typeof tfTypes.linspace = (...args) => ensureBackend().linspace(...args);
export const buffer: typeof tfTypes.buffer = (...args) => ensureBackend().buffer(...args);
export const oneHot: typeof tfTypes.oneHot = (...args) => ensureBackend().oneHot(...args);

// Math operations
export const add: typeof tfTypes.add = (...args) => ensureBackend().add(...args);
export const sub: typeof tfTypes.sub = (...args) => ensureBackend().sub(...args);
export const pow: typeof tfTypes.pow = (...args) => ensureBackend().pow(...args);
export const sqrt: typeof tfTypes.sqrt = (...args) => ensureBackend().sqrt(...args);
export const square: typeof tfTypes.square = (...args) => ensureBackend().square(...args);
export const maximum: typeof tfTypes.maximum = (...args) => ensureBackend().maximum(...args);
export const matMul: typeof tfTypes.matMul = (...args) => ensureBackend().matMul(...args);

// Reduction / selection
export const sum: typeof tfTypes.sum = (...args) => ensureBackend().sum(...args);
export const argMin: typeof tfTypes.argMin = (...args) => ensureBackend().argMin(...args);
export const gather: typeof tfTypes.gather = (...args) => ensureBackend().gather(...args);
export const topk: typeof tfTypes.topk = (...args) => ensureBackend().topk(...args);
export const scatterND: typeof tfTypes.scatterND = (...args) => ensureBackend().scatterND(...args);

// Tensor manipulation
export const slice: typeof tfTypes.slice = (...args) => ensureBackend().slice(...args);
export const concat: typeof tfTypes.concat = (...args) => ensureBackend().concat(...args);
export const stack: typeof tfTypes.stack = (...args) => ensureBackend().stack(...args);
export const cast: typeof tfTypes.cast = (...args) => ensureBackend().cast(...args);
export const expandDims: typeof tfTypes.expandDims = (...args) => ensureBackend().expandDims(...args);

// Comparison / logical
export const where: typeof tfTypes.where = (...args) => ensureBackend().where(...args);

// Memory management
export const tidy: typeof tfTypes.tidy = (...args) => ensureBackend().tidy(...args);
export const keep: typeof tfTypes.keep = (...args) => ensureBackend().keep(...args);
export const clone: typeof tfTypes.clone = (...args) => ensureBackend().clone(...args);
export const dispose: typeof tfTypes.dispose = (...args) => ensureBackend().dispose(...args);

// Random
export const randomUniform: typeof tfTypes.randomUniform = (...args) => ensureBackend().randomUniform(...args);
export const randomNormal: typeof tfTypes.randomNormal = (...args) => ensureBackend().randomNormal(...args);

// Backend utilities (used by benchmarks, init, and tests)
export const setBackend: typeof tfTypes.setBackend = (...args) => ensureBackend().setBackend(...args);
export const ready: typeof tfTypes.ready = () => ensureBackend().ready();
export const memory: typeof tfTypes.memory = () => ensureBackend().memory();
export const getBackend: typeof tfTypes.getBackend = () => ensureBackend().getBackend();
export const env: typeof tfTypes.env = () => ensureBackend().env();
export const engine: typeof tfTypes.engine = () => ensureBackend().engine();
export const disposeVariables: typeof tfTypes.disposeVariables = () => ensureBackend().disposeVariables();

// Namespace — linalg.qr() is used by eigen_qr.ts
export const linalg: typeof tfTypes.linalg = new Proxy({} as typeof tfTypes.linalg, {
  get(_target, prop: string | symbol) {
    const linalgNs = ensureBackend().linalg;
    return (linalgNs as Record<string | symbol, unknown>)[prop];
  }
});

// ---------------------------------------------------------------------------
// Default export — a Proxy that delegates all property access to ensureBackend()
// Used by code that accesses tf as a namespace object (e.g. tf.someFunction)
// ---------------------------------------------------------------------------
const tf: typeof tfTypes = new Proxy({} as typeof tfTypes, {
  get(_target, prop: string | symbol) {
    if (prop === '__esModule') return true;

    // For known named exports, return them directly to maintain identity.
    // Keep in sync with the named exports above.
    const namedExports: Record<string, unknown> = {
      tensor, tensor1d, tensor2d, tensor3d, scalar, zeros, ones, onesLike,
      fill, eye, linspace, buffer, oneHot, add, sub, pow, sqrt, square,
      maximum, matMul, sum, argMin, gather, topk, scatterND, slice, concat,
      stack, cast, expandDims, where, tidy, keep, clone, dispose,
      randomUniform, randomNormal, setBackend, ready, memory, getBackend,
      env, engine, disposeVariables, linalg,
    };

    if (typeof prop === 'string' && prop in namedExports) {
      return namedExports[prop];
    }

    // Fallback: delegate to the actual tf instance for anything else
    const instance = ensureBackend();
    const value = (instance as Record<string | symbol, unknown>)[prop];
    if (typeof value === 'function') {
      return (value as (...fnArgs: unknown[]) => unknown).bind(instance);
    }
    return value;
  }
});

export default tf;
