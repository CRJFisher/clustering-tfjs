/**
 * Browser-specific TensorFlow.js adapter
 * 
 * This module is used when building for browser environments.
 * It expects users to have loaded @tensorflow/tfjs separately.
 */

// Function to get tf from global
function getTf(): any {
  if (typeof window !== 'undefined' && (window as any).tf) {
    return (window as any).tf;
  }
  throw new Error('TensorFlow.js not found. Please load it before using this library.');
}

// Export functions that delegate to global tf
export const tensor = (...args: any[]) => getTf().tensor(...args);
export const tensor1d = (...args: any[]) => getTf().tensor1d(...args);
export const tensor2d = (...args: any[]) => getTf().tensor2d(...args);
export const tensor3d = (...args: any[]) => getTf().tensor3d(...args);
export const tensor4d = (...args: any[]) => getTf().tensor4d(...args);
export const add = (...args: any[]) => getTf().add(...args);
export const sub = (...args: any[]) => getTf().sub(...args);
export const mul = (...args: any[]) => getTf().mul(...args);
export const div = (...args: any[]) => getTf().div(...args);
export const matMul = (...args: any[]) => getTf().matMul(...args);
export const transpose = (...args: any[]) => getTf().transpose(...args);
export const mean = (...args: any[]) => getTf().mean(...args);
export const sum = (...args: any[]) => getTf().sum(...args);
export const sqrt = (...args: any[]) => getTf().sqrt(...args);
export const square = (...args: any[]) => getTf().square(...args);
export const norm = (...args: any[]) => getTf().norm(...args);
export const slice = (...args: any[]) => getTf().slice(...args);
export const concat = (...args: any[]) => getTf().concat(...args);
export const gather = (...args: any[]) => getTf().gather(...args);
export const unique = (...args: any[]) => getTf().unique(...args);
export const tidy = (...args: any[]) => getTf().tidy(...args);
export const dispose = (...args: any[]) => getTf().dispose(...args);
export const eye = (...args: any[]) => getTf().eye(...args);
export const diag = (...args: any[]) => getTf().diag(...args);
export const fill = (...args: any[]) => getTf().fill(...args);
export const stack = (...args: any[]) => getTf().stack(...args);
export const unstack = (...args: any[]) => getTf().unstack(...args);
export const split = (...args: any[]) => getTf().split(...args);
export const scalar = (...args: any[]) => getTf().scalar(...args);
export const keep = (...args: any[]) => getTf().keep(...args);
export const topk = (...args: any[]) => getTf().topk(...args);
export const scatterND = (...args: any[]) => getTf().scatterND(...args);
export const zeros = (...args: any[]) => getTf().zeros(...args);
export const ones = (...args: any[]) => getTf().ones(...args);
export const zerosLike = (...args: any[]) => getTf().zerosLike(...args);
export const onesLike = (...args: any[]) => getTf().onesLike(...args);
export const expandDims = (...args: any[]) => getTf().expandDims(...args);
export const maximum = (...args: any[]) => getTf().maximum(...args);
export const minimum = (...args: any[]) => getTf().minimum(...args);
export const pow = (...args: any[]) => getTf().pow(...args);
export const equal = (...args: any[]) => getTf().equal(...args);
export const greater = (...args: any[]) => getTf().greater(...args);
export const greaterEqual = (...args: any[]) => getTf().greaterEqual(...args);
export const less = (...args: any[]) => getTf().less(...args);
export const lessEqual = (...args: any[]) => getTf().lessEqual(...args);
export const logicalAnd = (...args: any[]) => getTf().logicalAnd(...args);
export const logicalOr = (...args: any[]) => getTf().logicalOr(...args);
export const logicalNot = (...args: any[]) => getTf().logicalNot(...args);
export const abs = (...args: any[]) => getTf().abs(...args);
export const neg = (...args: any[]) => getTf().neg(...args);
export const round = (...args: any[]) => getTf().round(...args);
export const ceil = (...args: any[]) => getTf().ceil(...args);
export const floor = (...args: any[]) => getTf().floor(...args);
export const sign = (...args: any[]) => getTf().sign(...args);
export const sin = (...args: any[]) => getTf().sin(...args);
export const cos = (...args: any[]) => getTf().cos(...args);
export const tan = (...args: any[]) => getTf().tan(...args);
export const asin = (...args: any[]) => getTf().asin(...args);
export const acos = (...args: any[]) => getTf().acos(...args);
export const atan = (...args: any[]) => getTf().atan(...args);
export const sinh = (...args: any[]) => getTf().sinh(...args);
export const cosh = (...args: any[]) => getTf().cosh(...args);
export const tanh = (...args: any[]) => getTf().tanh(...args);
export const elu = (...args: any[]) => getTf().elu(...args);
export const relu = (...args: any[]) => getTf().relu(...args);
export const selu = (...args: any[]) => getTf().selu(...args);
export const leakyRelu = (...args: any[]) => getTf().leakyRelu(...args);
export const prelu = (...args: any[]) => getTf().prelu(...args);
export const softmax = (...args: any[]) => getTf().softmax(...args);
export const image = () => getTf().image;
export const min = (...args: any[]) => getTf().min(...args);
export const max = (...args: any[]) => getTf().max(...args);
export const prod = (...args: any[]) => getTf().prod(...args);
export const cumsum = (...args: any[]) => getTf().cumsum(...args);
export const all = (...args: any[]) => getTf().all(...args);
export const any = (...args: any[]) => getTf().any(...args);
export const where = (...args: any[]) => getTf().where(...args);
export const argMax = (...args: any[]) => getTf().argMax(...args);
export const argMin = (...args: any[]) => getTf().argMin(...args);
export const memory = () => getTf().memory();
export const backend = () => getTf().backend();
export const env = () => getTf().env();
export const ready = () => getTf().ready();
export const setBackend = (...args: any[]) => getTf().setBackend(...args);
export const getBackend = () => getTf().getBackend();
export const Tensor = () => getTf().Tensor;

// Additional functions that might be needed
export const range = (...args: any[]) => getTf().range(...args);
export const linspace = (...args: any[]) => getTf().linspace(...args);
export const cast = (...args: any[]) => getTf().cast(...args);
export const squeeze = (...args: any[]) => getTf().squeeze(...args);
export const reshape = (...args: any[]) => getTf().reshape(...args);
export const reverse = (...args: any[]) => getTf().reverse(...args);
export const dot = (...args: any[]) => getTf().dot(...args);
export const outerProduct = (...args: any[]) => getTf().outerProduct(...args);
export const tensor5d = (...args: any[]) => getTf().tensor5d(...args);
export const tensor6d = (...args: any[]) => getTf().tensor6d(...args);
export const variable = (...args: any[]) => getTf().variable(...args);
export const grad = (...args: any[]) => getTf().grad(...args);
export const grads = (...args: any[]) => getTf().grads(...args);
export const customGrad = (...args: any[]) => getTf().customGrad(...args);
export const valueAndGrad = (...args: any[]) => getTf().valueAndGrad(...args);
export const valueAndGrads = (...args: any[]) => getTf().valueAndGrads(...args);
export const variableGrads = (...args: any[]) => getTf().variableGrads(...args);

// Default export
export default {
  // Creation
  tensor,
  tensor1d,
  tensor2d,
  tensor3d,
  tensor4d,
  tensor5d,
  tensor6d,
  variable,
  zeros,
  ones,
  zerosLike,
  onesLike,
  fill,
  scalar,
  range,
  linspace,
  
  // Manipulation
  concat,
  split,
  slice,
  stack,
  unstack,
  reverse,
  pad: (...args: any[]) => getTf().pad(...args),
  reshape,
  squeeze,
  expandDims,
  gather,
  
  // Math
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
  min,
  max,
  mean,
  sum,
  prod,
  norm,
  cumsum,
  argMax,
  argMin,
  
  // Logical
  equal,
  notEqual: (...args: any[]) => getTf().notEqual(...args),
  less,
  lessEqual,
  greater,
  greaterEqual,
  where,
  logicalAnd,
  logicalOr,
  logicalNot,
  logicalXor: (...args: any[]) => getTf().logicalXor(...args),
  
  // Trig
  sin,
  cos,
  tan,
  asin,
  acos,
  atan,
  sinh,
  cosh,
  tanh,
  
  // Linear algebra
  matMul,
  dot,
  outerProduct,
  transpose,
  
  // Utility
  tidy,
  dispose,
  keep,
  memory,
  backend,
  env,
  ready,
  setBackend,
  getBackend,
  
  // Special
  eye,
  diag,
  unique,
  maximum,
  minimum,
  softmax,
  elu,
  relu,
  selu,
  leakyRelu,
  prelu,
  sigmoid: (...args: any[]) => getTf().sigmoid(...args),
  log: (...args: any[]) => getTf().log(...args),
  exp: (...args: any[]) => getTf().exp(...args),
  
  // Tensors
  Tensor,
  
  // Other  
  cast,
  clone: (...args: any[]) => getTf().clone(...args),
  print: (...args: any[]) => getTf().print(...args),
  
  // Gradients
  grad,
  grads,
  customGrad,
  valueAndGrad,
  valueAndGrads,
  variableGrads,
  
  // Additional ops
  topk,
  scatterND,
  
  // Sub-namespaces (for compatibility)
  image,
  linalg: () => getTf().linalg,
  losses: () => getTf().losses,
  train: () => getTf().train,
  data: () => getTf().data,
  browser: () => getTf().browser,
  util: () => getTf().util,
  io: () => getTf().io,
  
  // Aliases
  batchNorm: (...args: any[]) => getTf().batchNorm(...args),
  localResponseNormalization: (...args: any[]) => getTf().localResponseNormalization(...args),
  separableConv2d: (...args: any[]) => getTf().separableConv2d(...args),
  depthwiseConv2d: (...args: any[]) => getTf().depthwiseConv2d(...args),
  conv1d: (...args: any[]) => getTf().conv1d(...args),
  conv2d: (...args: any[]) => getTf().conv2d(...args),
  conv2dTranspose: (...args: any[]) => getTf().conv2dTranspose(...args),
  conv3d: (...args: any[]) => getTf().conv3d(...args),
  conv3dTranspose: (...args: any[]) => getTf().conv3dTranspose(...args),
  maxPool: (...args: any[]) => getTf().maxPool(...args),
  avgPool: (...args: any[]) => getTf().avgPool(...args),
  pool: (...args: any[]) => getTf().pool(...args),
  maxPool3d: (...args: any[]) => getTf().maxPool3d(...args),
  avgPool3d: (...args: any[]) => getTf().avgPool3d(...args),
  
  // Complex
  complex: (...args: any[]) => getTf().complex(...args),
  real: (...args: any[]) => getTf().real(...args),
  imag: (...args: any[]) => getTf().imag(...args),
  
  // FFT
  fft: (...args: any[]) => getTf().fft(...args),
  ifft: (...args: any[]) => getTf().ifft(...args),
  rfft: (...args: any[]) => getTf().rfft(...args),
  irfft: (...args: any[]) => getTf().irfft(...args),
  
  // Boolean masks
  booleanMaskAsync: (...args: any[]) => getTf().booleanMaskAsync(...args),
  
  // RNG
  randomNormal: (...args: any[]) => getTf().randomNormal(...args),
  randomUniform: (...args: any[]) => getTf().randomUniform(...args),
  multinomial: (...args: any[]) => getTf().multinomial(...args),
  randomGamma: (...args: any[]) => getTf().randomGamma(...args),
};