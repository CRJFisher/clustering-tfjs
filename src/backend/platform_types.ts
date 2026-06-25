export type Platform = 'browser' | 'node' | 'react-native';

export type TensorFlowBackend =
  | 'cpu'        // All platforms
  | 'webgl'      // Browser
  | 'wasm'       // Browser
  | 'webgpu'     // Browser (experimental)
  | 'node'       // Node.js CPU
  | 'node-gpu'   // Node.js GPU (CUDA)
  | 'rn-webgl';  // React Native

export interface ReactNativeConfig {
  gl_implementation?: 'expo-gl' | 'gl-react-native' | 'auto';
  mobile_optimized?: boolean;
  warmup_iterations?: number;
}

export interface PlatformFeatures {
  gpu_acceleration: boolean;
  wasm_simd: boolean;
  node_bindings: boolean;
  webgl: boolean;
}
