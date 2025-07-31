/**
 * Jest setup file to handle cross-platform TensorFlow.js compatibility issues.
 * 
 * On Windows CI, @tensorflow/tfjs-node may fail to load due to native binding issues.
 * This setup file intercepts all imports of @tensorflow/tfjs-node and redirects them
 * to @tensorflow/tfjs when necessary.
 */

// Detect if we're on Windows or if tfjs-node fails to load
let tfModule;
let backendName;

try {
  // First try the Node.js backend (works on most platforms)
  tfModule = require('@tensorflow/tfjs-node');
  backendName = '@tensorflow/tfjs-node';
  console.log('✓ Using @tensorflow/tfjs-node backend for tests');
} catch (error) {
  console.warn('⚠️  @tensorflow/tfjs-node failed to load:', error.message);
  console.log('🔄 Falling back to @tensorflow/tfjs (CPU-only) backend');
  
  // Verify CPU-only version is available
  try {
    tfModule = require('@tensorflow/tfjs');
    backendName = '@tensorflow/tfjs';
    console.log('✓ @tensorflow/tfjs (CPU-only) backend is available');
  } catch (fallbackError) {
    console.error('❌ Failed to load any TensorFlow.js backend:', fallbackError.message);
    console.error('Please ensure either @tensorflow/tfjs-node or @tensorflow/tfjs is installed');
    process.exit(1);
  }
}

// Mock @tensorflow/tfjs-node to return the working backend
if (backendName === '@tensorflow/tfjs') {
  jest.mock('@tensorflow/tfjs-node', () => tfModule);
}

// Set environment variable for test files to know which backend to use
process.env.TF_FALLBACK_MODE = backendName === '@tensorflow/tfjs' ? 'true' : 'false';

// Set longer timeout for tests that might need to compile TensorFlow operations
jest.setTimeout(30000);