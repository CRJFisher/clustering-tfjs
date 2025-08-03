const { createDefaultPreset } = require("ts-jest");

const tsJestTransformCfg = createDefaultPreset().transform;

// On Windows CI, use moduleNameMapper to redirect tfjs-node to regular tfjs
const moduleNameMapper = {};
if (process.platform === 'win32' && process.env.CI) {
  console.log('Configuring Jest for Windows CI - redirecting @tensorflow/tfjs-node to @tensorflow/tfjs');
  moduleNameMapper['^@tensorflow/tfjs-node$'] = '@tensorflow/tfjs';
}

/** @type {import("jest").Config} **/
module.exports = {
  testEnvironment: "node",
  transform: {
    ...tsJestTransformCfg,
  },
  setupFiles: ["<rootDir>/test/setup.js"],
  moduleNameMapper,
};