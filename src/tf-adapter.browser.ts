/**
 * Browser-specific TensorFlow.js adapter
 * 
 * This module is used when building for browser environments.
 * It expects users to have loaded @tensorflow/tfjs separately.
 */

// For browser builds, webpack treats @tensorflow/tfjs as external
// But there seems to be an issue with how it's resolving
// Let's try accessing the global directly

declare global {
  interface Window {
    tf: typeof import('@tensorflow/tfjs');
  }
}

// Function to get tf from global
function getTf() {
  if (typeof window !== 'undefined' && window.tf) {
    return window.tf;
  }
  throw new Error('TensorFlow.js not found. Please load it before using this library.');
}

// Create namespace that delegates to global tf
const tfNamespace = {
  get tensor() { return getTf().tensor; },
  get tensor1d() { return getTf().tensor1d; },
  get tensor2d() { return getTf().tensor2d; },
  get tensor3d() { return getTf().tensor3d; },
  get tensor4d() { return getTf().tensor4d; },
  get tensor5d() { return getTf().tensor5d; },
  get tensor6d() { return getTf().tensor6d; },
  get ones() { return getTf().ones; },
  get zeros() { return getTf().zeros; },
  get fill() { return getTf().fill; },
  get linspace() { return getTf().linspace; },
  get range() { return getTf().range; },
  // Math operations
  get add() { return getTf().add; },
  get sub() { return getTf().sub; },
  get mul() { return getTf().mul; },
  get div() { return getTf().div; },
  get matMul() { return getTf().matMul; },
  get dot() { return getTf().dot; },
  get norm() { return getTf().norm; },
  get sum() { return getTf().sum; },
  get mean() { return getTf().mean; },
  get sqrt() { return getTf().sqrt; },
  get square() { return getTf().square; },
  get exp() { return getTf().exp; },
  get log() { return getTf().log; },
  get sigmoid() { return getTf().sigmoid; },
  get slice() { return getTf().slice; },
  get concat() { return getTf().concat; },
  get stack() { return getTf().stack; },
  get unstack() { return getTf().unstack; },
  get split() { return getTf().split; },
  get gather() { return getTf().gather; },
  get transpose() { return getTf().transpose; },
  get reverse() { return getTf().reverse; },
  get cast() { return getTf().cast; },
  get reshape() { return getTf().reshape; },
  get squeeze() { return getTf().squeeze; },
  get eye() { return getTf().eye; },
  get diag() { return getTf().diag; },
  get where() { return getTf().where; },
  get unique() { return getTf().unique; },
  get argMax() { return getTf().argMax; },
  get argMin() { return getTf().argMin; },
  // Utilities
  get tidy() { return getTf().tidy; },
  get dispose() { return getTf().dispose; },
  get memory() { return getTf().memory; },
  get backend() { return getTf().backend; },
  get env() { return getTf().env; },
  get ready() { return getTf().ready; },
  get setBackend() { return getTf().setBackend; },
  get getBackend() { return getTf().getBackend; },
  // Types
  get Tensor() { return getTf().Tensor; },
};

export default tfNamespace;
export const tensor = tfNamespace.tensor;
export const tensor1d = tfNamespace.tensor1d;
export const tensor2d = tfNamespace.tensor2d;
export const tensor3d = tfNamespace.tensor3d;
export const tensor4d = tfNamespace.tensor4d;
export const add = tfNamespace.add;
export const sub = tfNamespace.sub;
export const mul = tfNamespace.mul;
export const div = tfNamespace.div;
export const matMul = tfNamespace.matMul;
export const transpose = tfNamespace.transpose;
export const mean = tfNamespace.mean;
export const sum = tfNamespace.sum;
export const sqrt = tfNamespace.sqrt;
export const square = tfNamespace.square;
export const norm = tfNamespace.norm;
export const slice = tfNamespace.slice;
export const concat = tfNamespace.concat;
export const gather = tfNamespace.gather;
export const unique = tfNamespace.unique;
export const tidy = tfNamespace.tidy;
export const dispose = tfNamespace.dispose;
export const eye = tfNamespace.eye;
export const diag = tfNamespace.diag;
export const fill = tfNamespace.fill;
export const stack = tfNamespace.stack;
export const unstack = tfNamespace.unstack;
export const split = tfNamespace.split;