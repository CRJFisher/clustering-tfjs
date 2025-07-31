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

// Build ES Modules - in a temporary directory
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

// Build TypeScript declarations
console.log('Building TypeScript declarations...');
execSync('npx tsc -p tsconfig.types.json', { stdio: 'inherit' });

console.log('Build complete!');