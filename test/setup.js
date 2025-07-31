/**
 * Jest setup file to handle cross-platform TensorFlow.js compatibility issues.
 * 
 * On Windows CI, @tensorflow/tfjs-node may fail to load due to native binding issues.
 * This setup file provides a fallback to the CPU-only version for tests by setting
 * an environment variable that tests can check.
 */

// Detect if we're on Windows or if tfjs-node fails to load
let shouldUseFallback = false;

try {
  // First try the Node.js backend (works on most platforms)
  require('@tensorflow/tfjs-node');
  console.log('✓ Using @tensorflow/tfjs-node backend for tests');
} catch (error) {
  console.warn('⚠️  @tensorflow/tfjs-node failed to load:', error.message);
  console.log('🔄 Tests will fall back to @tensorflow/tfjs (CPU-only) backend when needed');
  
  // Verify CPU-only version is available
  try {
    require('@tensorflow/tfjs');
    console.log('✓ @tensorflow/tfjs (CPU-only) backend is available');
    shouldUseFallback = true;
  } catch (fallbackError) {
    console.error('❌ Failed to load any TensorFlow.js backend:', fallbackError.message);
    console.error('Please ensure either @tensorflow/tfjs-node or @tensorflow/tfjs is installed');
    process.exit(1);
  }
}

// Set environment variable for test files to know which backend to use
process.env.TF_FALLBACK_MODE = shouldUseFallback ? 'true' : 'false';

// Set longer timeout for tests that might need to compile TensorFlow operations
jest.setTimeout(30000);