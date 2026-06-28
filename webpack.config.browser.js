const path = require('path');
const TerserPlugin = require('terser-webpack-plugin');

module.exports = {
  mode: 'production',
  entry: './src/index.ts',
  target: 'web',
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'clustering.browser.js',
    library: {
      name: 'ClusteringTFJS',
      type: 'umd',
    },
    globalObject: 'this',
  },
  resolve: {
    extensions: ['.ts', '.js'],
    fallback: {
      // Browser doesn't have Node.js modules
      fs: false,
      path: false,
      crypto: false,
      os: false,
      util: false,
      assert: false,
      stream: false,
    },
  },
  module: {
    rules: [
      {
        test: /\.ts$/,
        use: 'ts-loader',
        exclude: /node_modules/,
      },
    ],
  },
  externals: {
    // TensorFlow.js should be loaded separately in browser
    '@tensorflow/tfjs': {
      commonjs: '@tensorflow/tfjs',
      commonjs2: '@tensorflow/tfjs',
      amd: '@tensorflow/tfjs',
      root: 'tf',
    },
    '@tensorflow/tfjs-core': {
      commonjs: '@tensorflow/tfjs-core',
      commonjs2: '@tensorflow/tfjs-core',
      amd: '@tensorflow/tfjs-core',
      root: 'tf',
    },
    // The per-backend dynamic imports in loader.browser.ts exist so a
    // code-splitting bundler (Vite, the demo site) emits one lazy chunk per
    // backend. A UMD library build cannot async-load chunks, so webpack would
    // otherwise inline every backend (~900KB) into the single UMD file. Script
    // -tag/CDN consumers of this bundle supply a full tfjs as window.tf — the
    // loader short-circuits to it and never imports a backend package — so the
    // backends belong outside this artifact, exactly like tfjs-core.
    '@tensorflow/tfjs-backend-cpu': {
      commonjs: '@tensorflow/tfjs-backend-cpu',
      commonjs2: '@tensorflow/tfjs-backend-cpu',
      amd: '@tensorflow/tfjs-backend-cpu',
      root: 'tf',
    },
    '@tensorflow/tfjs-backend-webgl': {
      commonjs: '@tensorflow/tfjs-backend-webgl',
      commonjs2: '@tensorflow/tfjs-backend-webgl',
      amd: '@tensorflow/tfjs-backend-webgl',
      root: 'tf',
    },
    '@tensorflow/tfjs-backend-webgpu': {
      commonjs: '@tensorflow/tfjs-backend-webgpu',
      commonjs2: '@tensorflow/tfjs-backend-webgpu',
      amd: '@tensorflow/tfjs-backend-webgpu',
      root: 'tf',
    },
    '@tensorflow/tfjs-backend-wasm': {
      commonjs: '@tensorflow/tfjs-backend-wasm',
      commonjs2: '@tensorflow/tfjs-backend-wasm',
      amd: '@tensorflow/tfjs-backend-wasm',
      root: 'tf',
    },
    // Exclude Node.js specific packages
    '@tensorflow/tfjs-node': 'empty',
    '@tensorflow/tfjs-node-gpu': 'empty',
  },
  optimization: {
    minimize: true,
    minimizer: [
      new TerserPlugin({
        terserOptions: {
          compress: {
            drop_console: true,
            passes: 2,
            pure_getters: true,
          },
          mangle: {
            properties: false,
          },
          format: {
            comments: false,
          },
        },
        extractComments: false,
      }),
    ],
  },
};