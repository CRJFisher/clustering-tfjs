/**
 * Browser-specific TensorFlow.js adapter
 * 
 * This module is used when building for browser environments.
 * It expects users to have loaded @tensorflow/tfjs separately.
 */

// Debug what we're getting from the import
import * as tfImport from '@tensorflow/tfjs';

// Log what we got (will be visible in build output)
if (typeof window !== 'undefined') {
  console.log('[tf-adapter.browser] tfImport:', tfImport);
  console.log('[tf-adapter.browser] tfImport.tensor2d:', tfImport.tensor2d);
  console.log('[tf-adapter.browser] window.tf:', (window as any).tf);
}

// In webpack browser build, @tensorflow/tfjs is marked as external and maps to global 'tf'
// However, there might be an issue with how it's being resolved
const tf = tfImport;

export default tf;
export * from '@tensorflow/tfjs';