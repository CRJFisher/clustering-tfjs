/**
 * Alternative browser adapter that uses global tf directly
 * This is a test to see if bypassing the import system fixes the issue
 */

// Type imports for proper typing
import type * as tfTypes from '@tensorflow/tfjs-core';

// Declare global window.tf
declare global {
  interface Window {
    tf: typeof tfTypes;
  }
}

// Get tf from global window object
const getGlobalTf = (): typeof tfTypes => {
  if (typeof window !== 'undefined' && window.tf) {
    return window.tf;
  }
  throw new Error('TensorFlow.js not found. Please load it before using this library.');
};

// Create a proxy that always gets the current global tf
const tfProxy = new Proxy({} as typeof tfTypes, {
  get(_target, prop: string) {
    const tf = getGlobalTf();
    return tf[prop as keyof typeof tfTypes];
  }
});

export default tfProxy;
export const tensor2d: typeof tfTypes.tensor2d = (...args) => getGlobalTf().tensor2d(...args);
export const tensor: typeof tfTypes.tensor = (...args) => getGlobalTf().tensor(...args);
export const tidy: typeof tfTypes.tidy = (...args) => getGlobalTf().tidy(...args);
export const dispose: typeof tfTypes.dispose = (...args) => getGlobalTf().dispose(...args);
// Add other commonly used exports...