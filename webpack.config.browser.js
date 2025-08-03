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
    alias: {
      // Use browser-specific adapter
      '../tf-adapter': path.resolve(__dirname, 'src/tf-adapter.browser.ts'),
      './tf-adapter': path.resolve(__dirname, 'src/tf-adapter.browser.ts'),
      // Replace Node.js loader with browser loader
      './tf-loader.node': path.resolve(__dirname, 'src/tf-loader.browser.ts'),
    },
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
            drop_console: false,
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