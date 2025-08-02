/**
 * Browser-specific TensorFlow.js adapter
 * 
 * This module is used when building for browser environments.
 * It expects users to have loaded @tensorflow/tfjs separately.
 */

import { getTensorFlow } from './tf-backend';

// Create a proxy that will get TensorFlow on demand
const tf = new Proxy({} as any, {
  get(target, prop) {
    const tfInstance = getTensorFlow();
    return tfInstance[prop as keyof typeof tfInstance];
  }
});

export default tf;
// Re-export types from core
export * from '@tensorflow/tfjs-core';