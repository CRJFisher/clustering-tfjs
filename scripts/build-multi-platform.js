#!/usr/bin/env node

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

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

// Build ES Modules
console.log('Building ES Modules...');
fs.mkdirSync('dist-esm', { recursive: true });
execSync('npx tsc -p tsconfig.esm.json --outDir dist-esm', { stdio: 'inherit' });

// Copy ESM files with .esm.js extension
const esmFiles = fs.readdirSync('dist-esm');
esmFiles.forEach(file => {
  if (file.endsWith('.js')) {
    const content = fs.readFileSync(path.join('dist-esm', file), 'utf8');
    const newFileName = file.replace('.js', '.esm.js');
    fs.writeFileSync(path.join('dist', newFileName), content);
  }
});

// Clean temporary ESM directory
fs.rmSync('dist-esm', { recursive: true });

// Build browser bundle with webpack
console.log('Building browser bundle...');
execSync('npx webpack --config webpack.config.browser.js', { stdio: 'inherit' });

// Build Node.js optimized bundle with webpack
console.log('Building Node.js bundle...');
execSync('npx webpack --config webpack.config.node.js', { stdio: 'inherit' });

console.log('Multi-platform build complete!');
console.log('Output files:');
console.log('  - dist/index.js (CommonJS)');
console.log('  - dist/index.esm.js (ES Module)');
console.log('  - dist/clustering.browser.js (Browser UMD)');
console.log('  - dist/clustering.node.js (Node.js optimized)');
console.log('  - dist/index.d.ts (TypeScript definitions)');