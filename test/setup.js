/**
 * Jest setup file to handle cross-platform TensorFlow.js compatibility issues.
 * 
 * For tests, we pre-load the TensorFlow.js backend to ensure compatibility.
 */

// Try to load TensorFlow.js Node backend
try {
  require('@tensorflow/tfjs-node');
  console.log('✓ Using @tensorflow/tfjs-node backend for tests');
} catch (error) {
  console.warn('⚠️  @tensorflow/tfjs-node failed to load:', error.message);
  console.log('🔄 Tests will use fallback backend');
}

// Set longer timeout for tests that might need to compile TensorFlow operations
jest.setTimeout(30000);