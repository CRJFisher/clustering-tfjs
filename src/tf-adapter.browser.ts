/**
 * Browser-specific TensorFlow.js adapter
 * 
 * This module is used when building for browser environments.
 * It expects users to have loaded @tensorflow/tfjs separately.
 */

// Type imports for proper typing
import type * as tfTypes from '@tensorflow/tfjs-core';

// Declare global window.tf
declare global {
  interface Window {
    tf: typeof tfTypes;
  }
}

// Function to get tf from global
function getTf(): typeof tfTypes {
  if (typeof window !== 'undefined' && window.tf) {
    return window.tf;
  }
  throw new Error('TensorFlow.js not found. Please load it before using this library.');
}

// Re-export all tf functions, properly typed
const tf = new Proxy({} as typeof tfTypes, {
  get(_target, prop: string) {
    const tfInstance = getTf();
    return tfInstance[prop as keyof typeof tfTypes];
  }
});

// Export commonly used functions for better tree-shaking
export const tensor: typeof tfTypes.tensor = (...args) => tf.tensor(...args);
export const tensor1d: typeof tfTypes.tensor1d = (...args) => tf.tensor1d(...args);
export const tensor2d: typeof tfTypes.tensor2d = (...args) => tf.tensor2d(...args);
export const tensor3d: typeof tfTypes.tensor3d = (...args) => tf.tensor3d(...args);
export const tensor4d: typeof tfTypes.tensor4d = (...args) => tf.tensor4d(...args);
export const tensor5d: typeof tfTypes.tensor5d = (...args) => tf.tensor5d(...args);
export const tensor6d: typeof tfTypes.tensor6d = (...args) => tf.tensor6d(...args);
export const variable: typeof tfTypes.variable = (...args) => tf.variable(...args);
export const scalar: typeof tfTypes.scalar = (...args) => tf.scalar(...args);
export const zeros: typeof tfTypes.zeros = (...args) => tf.zeros(...args);
export const ones: typeof tfTypes.ones = (...args) => tf.ones(...args);
export const zerosLike: typeof tfTypes.zerosLike = (...args) => tf.zerosLike(...args);
export const onesLike: typeof tfTypes.onesLike = (...args) => tf.onesLike(...args);
export const fill: typeof tfTypes.fill = (...args) => tf.fill(...args);
export const range: typeof tfTypes.range = (...args) => tf.range(...args);
export const linspace: typeof tfTypes.linspace = (...args) => tf.linspace(...args);

// Math operations
export const add: typeof tfTypes.add = (...args) => tf.add(...args);
export const sub: typeof tfTypes.sub = (...args) => tf.sub(...args);
export const mul: typeof tfTypes.mul = (...args) => tf.mul(...args);
export const div: typeof tfTypes.div = (...args) => tf.div(...args);
export const pow: typeof tfTypes.pow = (...args) => tf.pow(...args);
export const sqrt: typeof tfTypes.sqrt = (...args) => tf.sqrt(...args);
export const square: typeof tfTypes.square = (...args) => tf.square(...args);
export const abs: typeof tfTypes.abs = (...args) => tf.abs(...args);
export const neg: typeof tfTypes.neg = (...args) => tf.neg(...args);
export const sign: typeof tfTypes.sign = (...args) => tf.sign(...args);
export const round: typeof tfTypes.round = (...args) => tf.round(...args);
export const floor: typeof tfTypes.floor = (...args) => tf.floor(...args);
export const ceil: typeof tfTypes.ceil = (...args) => tf.ceil(...args);
export const sin: typeof tfTypes.sin = (...args) => tf.sin(...args);
export const cos: typeof tfTypes.cos = (...args) => tf.cos(...args);
export const tan: typeof tfTypes.tan = (...args) => tf.tan(...args);
export const asin: typeof tfTypes.asin = (...args) => tf.asin(...args);
export const acos: typeof tfTypes.acos = (...args) => tf.acos(...args);
export const atan: typeof tfTypes.atan = (...args) => tf.atan(...args);
export const sinh: typeof tfTypes.sinh = (...args) => tf.sinh(...args);
export const cosh: typeof tfTypes.cosh = (...args) => tf.cosh(...args);
export const tanh: typeof tfTypes.tanh = (...args) => tf.tanh(...args);
export const elu: typeof tfTypes.elu = (...args) => tf.elu(...args);
export const relu: typeof tfTypes.relu = (...args) => tf.relu(...args);
export const selu: typeof tfTypes.selu = (...args) => tf.selu(...args);
export const leakyRelu: typeof tfTypes.leakyRelu = (...args) => tf.leakyRelu(...args);
export const prelu: typeof tfTypes.prelu = (...args) => tf.prelu(...args);
export const softmax: typeof tfTypes.softmax = (...args) => tf.softmax(...args);

// Linear algebra
export const matMul: typeof tfTypes.matMul = (...args) => tf.matMul(...args);
export const dot: typeof tfTypes.dot = (...args) => tf.dot(...args);
export const outerProduct: typeof tfTypes.outerProduct = (...args) => tf.outerProduct(...args);
export const transpose: typeof tfTypes.transpose = (...args) => tf.transpose(...args);
export const norm: typeof tfTypes.norm = (...args) => tf.norm(...args);

// Reduction
export const mean: typeof tfTypes.mean = (...args) => tf.mean(...args);
export const sum: typeof tfTypes.sum = (...args) => tf.sum(...args);
export const min: typeof tfTypes.min = (...args) => tf.min(...args);
export const max: typeof tfTypes.max = (...args) => tf.max(...args);
export const prod: typeof tfTypes.prod = (...args) => tf.prod(...args);
export const cumsum: typeof tfTypes.cumsum = (...args) => tf.cumsum(...args);
export const all: typeof tfTypes.all = (...args) => tf.all(...args);
export const any: typeof tfTypes.any = (...args) => tf.any(...args);
export const argMax: typeof tfTypes.argMax = (...args) => tf.argMax(...args);
export const argMin: typeof tfTypes.argMin = (...args) => tf.argMin(...args);

// Manipulation
export const slice: typeof tfTypes.slice = (...args) => tf.slice(...args);
export const concat: typeof tfTypes.concat = (...args) => tf.concat(...args);
export const stack: typeof tfTypes.stack = (...args) => tf.stack(...args);
export const unstack: typeof tfTypes.unstack = (...args) => tf.unstack(...args);
export const split: typeof tfTypes.split = (...args) => tf.split(...args);
export const gather: typeof tfTypes.gather = (...args) => tf.gather(...args);
export const reverse: typeof tfTypes.reverse = (...args) => tf.reverse(...args);
export const cast: typeof tfTypes.cast = (...args) => tf.cast(...args);
export const reshape: typeof tfTypes.reshape = (...args) => tf.reshape(...args);
export const squeeze: typeof tfTypes.squeeze = (...args) => tf.squeeze(...args);
export const expandDims: typeof tfTypes.expandDims = (...args) => tf.expandDims(...args);

// Logical
export const equal: typeof tfTypes.equal = (...args) => tf.equal(...args);
export const greater: typeof tfTypes.greater = (...args) => tf.greater(...args);
export const greaterEqual: typeof tfTypes.greaterEqual = (...args) => tf.greaterEqual(...args);
export const less: typeof tfTypes.less = (...args) => tf.less(...args);
export const lessEqual: typeof tfTypes.lessEqual = (...args) => tf.lessEqual(...args);
export const logicalAnd: typeof tfTypes.logicalAnd = (...args) => tf.logicalAnd(...args);
export const logicalOr: typeof tfTypes.logicalOr = (...args) => tf.logicalOr(...args);
export const logicalNot: typeof tfTypes.logicalNot = (...args) => tf.logicalNot(...args);
export const where: typeof tfTypes.where = (...args) => tf.where(...args);

// Special tensors
export const eye: typeof tfTypes.eye = (...args) => tf.eye(...args);
export const diag: typeof tfTypes.diag = (...args) => tf.diag(...args);
export const unique: typeof tfTypes.unique = (...args) => tf.unique(...args);

// Utility
export const tidy: typeof tfTypes.tidy = (...args) => tf.tidy(...args);
export const dispose: typeof tfTypes.dispose = (...args) => tf.dispose(...args);
export const keep: typeof tfTypes.keep = (...args) => tf.keep(...args);
export const memory: typeof tfTypes.memory = () => tf.memory();
export const backend: typeof tfTypes.backend = () => tf.backend();
export const env: typeof tfTypes.env = () => tf.env();
export const ready: typeof tfTypes.ready = () => tf.ready();
export const setBackend: typeof tfTypes.setBackend = (...args) => tf.setBackend(...args);
export const getBackend: typeof tfTypes.getBackend = () => tf.getBackend();

// Advanced
export const grad: typeof tfTypes.grad = (...args) => tf.grad(...args);
export const grads: typeof tfTypes.grads = (...args) => tf.grads(...args);
export const customGrad: typeof tfTypes.customGrad = (...args) => tf.customGrad(...args);
export const valueAndGrad: typeof tfTypes.valueAndGrad = (...args) => tf.valueAndGrad(...args);
export const valueAndGrads: typeof tfTypes.valueAndGrads = (...args) => tf.valueAndGrads(...args);
export const variableGrads: typeof tfTypes.variableGrads = (...args) => tf.variableGrads(...args);

// Scatter
export const topk: typeof tfTypes.topk = (...args) => tf.topk(...args);
export const scatterND: typeof tfTypes.scatterND = (...args) => tf.scatterND(...args);

// Globals/Types - Access from runtime tf object
export const Tensor = () => getTf().Tensor;

// Namespaces - return functions to avoid immediate evaluation
export const image = () => getTf().image;
export const linalg = () => getTf().linalg;
export const losses = () => getTf().losses;
export const train = () => getTf().train;
// data namespace is not in @tensorflow/tfjs-core, only in full tfjs
// Return type is unknown since data namespace types aren't in core
export const data = (): unknown => {
  const tfInstance = getTf();
  if ('data' in tfInstance) {
    return (tfInstance as { data: unknown }).data;
  }
  throw new Error('TensorFlow.js data API not available. Please load @tensorflow/tfjs instead of @tensorflow/tfjs-core');
};
export const browser = () => getTf().browser;
export const util = () => getTf().util;
export const io = () => getTf().io;

// Additional functions that might be needed - use the proxy
const sigmoid: typeof tfTypes.sigmoid = (...args) => tf.sigmoid(...args);
const log: typeof tfTypes.log = (...args) => tf.log(...args);
const exp: typeof tfTypes.exp = (...args) => tf.exp(...args);
const maximum: typeof tfTypes.maximum = (...args) => tf.maximum(...args);
const minimum: typeof tfTypes.minimum = (...args) => tf.minimum(...args);
const clone: typeof tfTypes.clone = (...args) => tf.clone(...args);
const print: typeof tfTypes.print = (...args) => tf.print(...args);
const pad: typeof tfTypes.pad = (...args) => tf.pad(...args);
const notEqual: typeof tfTypes.notEqual = (...args) => tf.notEqual(...args);
const logicalXor: typeof tfTypes.logicalXor = (...args) => tf.logicalXor(...args);
const batchNorm: typeof tfTypes.batchNorm = (...args) => tf.batchNorm(...args);
const localResponseNormalization: typeof tfTypes.localResponseNormalization = (...args) => tf.localResponseNormalization(...args);
const separableConv2d: typeof tfTypes.separableConv2d = (...args) => tf.separableConv2d(...args);
const depthwiseConv2d: typeof tfTypes.depthwiseConv2d = (...args) => tf.depthwiseConv2d(...args);
const conv1d: typeof tfTypes.conv1d = (...args) => tf.conv1d(...args);
const conv2d: typeof tfTypes.conv2d = (...args) => tf.conv2d(...args);
const conv2dTranspose: typeof tfTypes.conv2dTranspose = (...args) => tf.conv2dTranspose(...args);
const conv3d: typeof tfTypes.conv3d = (...args) => tf.conv3d(...args);
const conv3dTranspose: typeof tfTypes.conv3dTranspose = (...args) => tf.conv3dTranspose(...args);
const maxPool: typeof tfTypes.maxPool = (...args) => tf.maxPool(...args);
const avgPool: typeof tfTypes.avgPool = (...args) => tf.avgPool(...args);
const pool: typeof tfTypes.pool = (...args) => tf.pool(...args);
const maxPool3d: typeof tfTypes.maxPool3d = (...args) => tf.maxPool3d(...args);
const avgPool3d: typeof tfTypes.avgPool3d = (...args) => tf.avgPool3d(...args);
const complex: typeof tfTypes.complex = (...args) => tf.complex(...args);
const real: typeof tfTypes.real = (...args) => tf.real(...args);
const imag: typeof tfTypes.imag = (...args) => tf.imag(...args);
const fft: typeof tfTypes.fft = (...args) => tf.fft(...args);
const ifft: typeof tfTypes.ifft = (...args) => tf.ifft(...args);
const rfft: typeof tfTypes.rfft = (...args) => tf.rfft(...args);
const irfft: typeof tfTypes.irfft = (...args) => tf.irfft(...args);
const booleanMaskAsync: typeof tfTypes.booleanMaskAsync = (...args) => tf.booleanMaskAsync(...args);
const randomNormal: typeof tfTypes.randomNormal = (...args) => tf.randomNormal(...args);
const randomUniform: typeof tfTypes.randomUniform = (...args) => tf.randomUniform(...args);
const multinomial: typeof tfTypes.multinomial = (...args) => tf.multinomial(...args);
const randomGamma: typeof tfTypes.randomGamma = (...args) => tf.randomGamma(...args);

// Default export as namespace
export default {
  // Export all our typed functions
  tensor,
  tensor1d,
  tensor2d,
  tensor3d,
  tensor4d,
  tensor5d,
  tensor6d,
  variable,
  scalar,
  zeros,
  ones,
  zerosLike,
  onesLike,
  fill,
  range,
  linspace,
  add,
  sub,
  mul,
  div,
  pow,
  sqrt,
  square,
  abs,
  neg,
  sign,
  round,
  floor,
  ceil,
  sin,
  cos,
  tan,
  asin,
  acos,
  atan,
  sinh,
  cosh,
  tanh,
  elu,
  relu,
  selu,
  leakyRelu,
  prelu,
  softmax,
  matMul,
  dot,
  outerProduct,
  transpose,
  norm,
  mean,
  sum,
  min,
  max,
  prod,
  cumsum,
  all,
  any,
  argMax,
  argMin,
  slice,
  concat,
  stack,
  unstack,
  split,
  gather,
  reverse,
  cast,
  reshape,
  squeeze,
  expandDims,
  equal,
  greater,
  greaterEqual,
  less,
  lessEqual,
  logicalAnd,
  logicalOr,
  logicalNot,
  where,
  eye,
  diag,
  unique,
  tidy,
  dispose,
  keep,
  memory,
  backend,
  env,
  ready,
  setBackend,
  getBackend,
  grad,
  grads,
  customGrad,
  valueAndGrad,
  valueAndGrads,
  variableGrads,
  topk,
  scatterND,
  Tensor,
  image,
  linalg,
  losses,
  train,
  data,
  browser,
  util,
  io,
  // Additional
  sigmoid,
  log,
  exp,
  maximum,
  minimum,
  clone,
  print,
  pad,
  notEqual,
  logicalXor,
  batchNorm,
  localResponseNormalization,
  separableConv2d,
  depthwiseConv2d,
  conv1d,
  conv2d,
  conv2dTranspose,
  conv3d,
  conv3dTranspose,
  maxPool,
  avgPool,
  pool,
  maxPool3d,
  avgPool3d,
  complex,
  real,
  imag,
  fft,
  ifft,
  rfft,
  irfft,
  booleanMaskAsync,
  randomNormal,
  randomUniform,
  multinomial,
  randomGamma,
};