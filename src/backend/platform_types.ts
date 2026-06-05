/**
 * Platform detection and configuration types for clustering-js
 */

/**
 * Supported platform environments
 */
export type Platform = 'browser' | 'node' | 'react-native';

/**
 * TensorFlow.js backend options by platform
 */
export type TensorFlowBackend = 
  | 'cpu'        // All platforms
  | 'webgl'      // Browser
  | 'wasm'       // Browser
  | 'webgpu'     // Browser (experimental)
  | 'node'       // Node.js CPU
  | 'node-gpu'   // Node.js GPU (CUDA)
  | 'rn-webgl';  // React Native

/**
 * React Native specific configuration
 */
export interface ReactNativeConfig {
  /**
   * Preferred GL implementation
   * - 'expo-gl': For Expo managed apps
   * - 'gl-react-native': For bare React Native apps
   * - 'auto': Auto-detect (default)
   */
  gl_implementation?: 'expo-gl' | 'gl-react-native' | 'auto';
  
  /**
   * Enable mobile-optimized settings
   * - Uses float32 by default
   * - Enables memory management optimizations
   * - Configures smaller batch sizes
   */
  mobile_optimized?: boolean;
  
  /**
   * Warmup iterations for TensorFlow.js graph compilation
   * Higher values may improve performance but increase initialization time
   */
  warmup_iterations?: number;
}

/**
 * Extended backend configuration with React Native support
 */
export interface ExtendedBackendConfig {
  /**
   * Preferred backend to use. If not specified, will auto-detect.
   */
  backend?: TensorFlowBackend;
  
  /**
   * Custom flags to pass to the TensorFlow.js backend
   */
  flags?: Record<string, unknown>;
  
  /**
   * React Native specific configuration
   */
  react_native?: ReactNativeConfig;
  
  /**
   * Force a specific platform detection (for testing)
   */
  force_platform?: Platform;
}

/**
 * Backend-specific features available in different environments
 */
export interface PlatformFeatures {
  gpu_acceleration: boolean;
  wasm_simd: boolean;
  node_bindings: boolean;
  webgl: boolean;
}

/**
 * Platform detection result
 */
export interface PlatformInfo {
  platform: Platform;
  available_backends: TensorFlowBackend[];
  recommended_backend: TensorFlowBackend;
  is_gpu_available: boolean;
}