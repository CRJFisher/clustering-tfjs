import type * as tf_types from '@tensorflow/tfjs-core';
import { ensure_backend } from './backend';

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

export const tensor: typeof tf_types.tensor = (...args) => ensure_backend().tensor(...args);
export const tensor1d: typeof tf_types.tensor1d = (...args) => ensure_backend().tensor1d(...args);
export const tensor2d: typeof tf_types.tensor2d = (...args) => ensure_backend().tensor2d(...args);
export const tensor3d: typeof tf_types.tensor3d = (...args) => ensure_backend().tensor3d(...args);
export const scalar: typeof tf_types.scalar = (...args) => ensure_backend().scalar(...args);
export const zeros: typeof tf_types.zeros = (...args) => ensure_backend().zeros(...args);
export const ones: typeof tf_types.ones = (...args) => ensure_backend().ones(...args);
export const ones_like: typeof tf_types.onesLike = (...args) => ensure_backend().onesLike(...args);
export const fill: typeof tf_types.fill = (...args) => ensure_backend().fill(...args);
export const eye: typeof tf_types.eye = (...args) => ensure_backend().eye(...args);
export const linspace: typeof tf_types.linspace = (...args) => ensure_backend().linspace(...args);
export const buffer: typeof tf_types.buffer = (...args) => ensure_backend().buffer(...args);
export const one_hot: typeof tf_types.oneHot = (...args) => ensure_backend().oneHot(...args);
export const add: typeof tf_types.add = (...args) => ensure_backend().add(...args);
export const sub: typeof tf_types.sub = (...args) => ensure_backend().sub(...args);
export const pow: typeof tf_types.pow = (...args) => ensure_backend().pow(...args);
export const sqrt: typeof tf_types.sqrt = (...args) => ensure_backend().sqrt(...args);
export const square: typeof tf_types.square = (...args) => ensure_backend().square(...args);
export const maximum: typeof tf_types.maximum = (...args) => ensure_backend().maximum(...args);
export const mat_mul: typeof tf_types.matMul = (...args) => ensure_backend().matMul(...args);
export const sum: typeof tf_types.sum = (...args) => ensure_backend().sum(...args);
export const arg_min: typeof tf_types.argMin = (...args) => ensure_backend().argMin(...args);
export const gather: typeof tf_types.gather = (...args) => ensure_backend().gather(...args);
export const topk: typeof tf_types.topk = (...args) => ensure_backend().topk(...args);
export const scatter_nd: typeof tf_types.scatterND = (...args) => ensure_backend().scatterND(...args);
export const slice: typeof tf_types.slice = (...args) => ensure_backend().slice(...args);
export const concat: typeof tf_types.concat = (...args) => ensure_backend().concat(...args);
export const stack: typeof tf_types.stack = (...args) => ensure_backend().stack(...args);
export const cast: typeof tf_types.cast = (...args) => ensure_backend().cast(...args);
export const expand_dims: typeof tf_types.expandDims = (...args) => ensure_backend().expandDims(...args);
export const where: typeof tf_types.where = (...args) => ensure_backend().where(...args);
export const tidy: typeof tf_types.tidy = (...args) => ensure_backend().tidy(...args);
export const keep: typeof tf_types.keep = (...args) => ensure_backend().keep(...args);
export const clone: typeof tf_types.clone = (...args) => ensure_backend().clone(...args);
export const dispose: typeof tf_types.dispose = (...args) => ensure_backend().dispose(...args);
export const random_uniform: typeof tf_types.randomUniform = (...args) => ensure_backend().randomUniform(...args);
export const random_normal: typeof tf_types.randomNormal = (...args) => ensure_backend().randomNormal(...args);
export const set_backend: typeof tf_types.setBackend = (...args) => ensure_backend().setBackend(...args);
export const ready: typeof tf_types.ready = () => ensure_backend().ready();
export const memory: typeof tf_types.memory = () => ensure_backend().memory();
export const get_backend: typeof tf_types.getBackend = () => ensure_backend().getBackend();
export const env: typeof tf_types.env = () => ensure_backend().env();
export const engine: typeof tf_types.engine = () => ensure_backend().engine();
export const dispose_variables: typeof tf_types.disposeVariables = () => ensure_backend().disposeVariables();

// Namespace — linalg.qr() is used by eigen_qr.ts
export const linalg: typeof tf_types.linalg = new Proxy({} as typeof tf_types.linalg, {
  get(_target, prop: string | symbol) {
    const linalg_ns = ensure_backend().linalg;
    return (linalg_ns as Record<string | symbol, unknown>)[prop];
  }
});

const tf: typeof tf_types = new Proxy({} as typeof tf_types, {
  get(_target, prop: string | symbol) {
    if (prop === '__esModule') return true;

    // Return named exports directly so callers get stable function identity.
    // Keep in sync with the named exports above.
    const named_exports: Record<string, unknown> = {
      tensor, tensor1d, tensor2d, tensor3d, scalar, zeros, ones, ones_like,
      fill, eye, linspace, buffer, one_hot, add, sub, pow, sqrt, square,
      maximum, mat_mul, sum, arg_min, gather, topk, scatter_nd, slice, concat,
      stack, cast, expand_dims, where, tidy, keep, clone, dispose,
      random_uniform, random_normal, set_backend, ready, memory, get_backend,
      env, engine, dispose_variables, linalg,
    };

    if (typeof prop === 'string' && prop in named_exports) {
      return named_exports[prop];
    }

    const instance = ensure_backend();
    const value = (instance as Record<string | symbol, unknown>)[prop];
    if (typeof value === 'function') {
      return (value as (...fn_args: unknown[]) => unknown).bind(instance);
    }
    return value;
  }
});

export default tf;
