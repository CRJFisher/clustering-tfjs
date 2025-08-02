/**
 * Browser-specific TensorFlow.js adapter
 * 
 * This module is used when building for browser environments.
 * It expects users to have loaded @tensorflow/tfjs separately.
 */

// In the browser build, webpack is configured to treat @tensorflow/tfjs as external
// and map it to the global 'tf' object (see webpack.config.browser.js externals)
// So we can directly import and re-export it

import * as tf from '@tensorflow/tfjs';

export default tf;
export * from '@tensorflow/tfjs';