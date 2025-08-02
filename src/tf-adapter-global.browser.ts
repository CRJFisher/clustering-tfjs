/**
 * Alternative browser adapter that uses global tf directly
 * This is a test to see if bypassing the import system fixes the issue
 */

// Get tf from global window object
const getGlobalTf = () => {
  if (typeof window !== 'undefined' && (window as any).tf) {
    return (window as any).tf;
  }
  throw new Error('TensorFlow.js not found. Please load it before using this library.');
};

// Create a proxy that always gets the current global tf
const tfProxy = new Proxy({}, {
  get(target, prop) {
    const tf = getGlobalTf();
    return tf[prop];
  }
});

export default tfProxy;
export const tensor2d = (...args: any[]) => getGlobalTf().tensor2d(...args);
export const tensor = (...args: any[]) => getGlobalTf().tensor(...args);
export const tidy = (fn: any) => getGlobalTf().tidy(fn);
export const dispose = (tensor: any) => getGlobalTf().dispose(tensor);
// Add other commonly used exports...