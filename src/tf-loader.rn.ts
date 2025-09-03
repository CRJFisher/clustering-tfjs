/**
 * React Native-specific TensorFlow.js loader
 * 
 * This module loads TensorFlow.js for React Native environments
 * with rn-webgl backend and CPU fallback.
 * 
 * Users must install:
 * - @tensorflow/tfjs-react-native
 * - @tensorflow/tfjs-react-native-platform (for Expo)
 * - expo-gl and expo-gl-cpp (for Expo)
 * OR
 * - gl-react-native (for bare React Native)
 */

export async function loadTensorFlow() {
  try {
    // Dynamic import to avoid build-time dependency
    // The actual module name is passed as a string to bypass TypeScript checking
    const tfRNModule = '@tensorflow/tfjs-react-native';
    const tf = await import(/* webpackIgnore: true */ tfRNModule as string) as typeof import('@tensorflow/tfjs');
    
    // Wait for TensorFlow.js to initialize
    await tf.ready();
    
    // Try to set rn-webgl backend for GPU acceleration
    try {
      await tf.setBackend('rn-webgl');
      console.log('Using TensorFlow.js React Native WebGL backend (GPU accelerated)');
    } catch (webglError) {
      // Fallback to CPU backend if WebGL not available
      console.warn('WebGL backend not available, falling back to CPU');
      await tf.setBackend('cpu');
      console.log('Using TensorFlow.js CPU backend');
    }
    
    return tf;
  } catch (error) {
    throw new Error(
      'TensorFlow.js React Native not found. Please install:\n' +
      '- @tensorflow/tfjs-react-native\n' +
      '- Platform-specific GL dependencies:\n' +
      '  For Expo: expo-gl and expo-gl-cpp\n' +
      '  For bare RN: gl-react-native\n\n' +
      'Also ensure you have called tf.ready() before using the library.'
    );
  }
}