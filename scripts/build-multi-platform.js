#!/usr/bin/env node

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// Get target from command line arguments
const target = process.argv[2]; // 'browser', 'node', or undefined for all

// Clean dist directory
console.log('Cleaning dist directory...');
if (fs.existsSync('dist')) {
  fs.rmSync('dist', { recursive: true });
}
fs.mkdirSync('dist', { recursive: true });

// Build TypeScript declarations first
console.log('Building TypeScript declarations...');
execSync('npx tsc -p tsconfig.types.json', { stdio: 'inherit' });

// Build CommonJS (Node.js default)
console.log('Building CommonJS...');
execSync('npx tsc -p tsconfig.cjs.json', { stdio: 'inherit' });

// Build ES Modules into dist/esm/
console.log('Building ES Modules...');
execSync('npx tsc -p tsconfig.esm.json --outDir dist/esm', { stdio: 'inherit' });

// Create ESM entry point that re-exports from the esm/ subdirectory
fs.writeFileSync(
  path.join('dist', 'index.esm.js'),
  "export * from './esm/index.js';\n"
);

// Build specific target or all
if (!target || target === 'browser') {
  // Build browser bundle with webpack
  console.log('Building browser bundle...');
  execSync('npx webpack --config webpack.config.browser.js', { stdio: 'inherit' });
}

if (!target || target === 'node') {
  // Build Node.js optimized bundle with webpack
  console.log('Building Node.js bundle...');
  execSync('npx webpack --config webpack.config.node.js', { stdio: 'inherit' });
}

console.log(`${target ? target.charAt(0).toUpperCase() + target.slice(1) : 'Multi-platform'} build complete!`);
console.log('Output files:');
console.log('  - dist/index.js (CommonJS)');
console.log('  - dist/index.esm.js (ES Module)');
console.log('  - dist/clustering.browser.js (Browser UMD)');
console.log('  - dist/clustering.node.js (Node.js optimized)');
console.log('  - dist/index.d.ts (TypeScript definitions)');