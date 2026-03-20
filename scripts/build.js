#!/usr/bin/env node

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// Clean dist directory
console.log('Cleaning dist directory...');
if (fs.existsSync('dist')) {
  fs.rmSync('dist', { recursive: true });
}

// Build CommonJS
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

// Build TypeScript declarations
console.log('Building TypeScript declarations...');
execSync('npx tsc -p tsconfig.types.json', { stdio: 'inherit' });

console.log('Build complete!');