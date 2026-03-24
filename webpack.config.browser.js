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