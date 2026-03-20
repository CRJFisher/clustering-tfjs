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

// Copy ESM files with .esm.js extension (recursive)
function copyEsmFiles(srcDir, destDir) {
  const entries = fs.readdirSync(srcDir, { withFileTypes: true });
  for (const entry of entries) {
    const srcPath = path.join(srcDir, entry.name);
    const destPath = path.join(destDir, entry.name);
    if (entry.isDirectory()) {
      fs.mkdirSync(destPath, { recursive: true });
      copyEsmFiles(srcPath, destPath);
    } else if (entry.name.endsWith('.js')) {
      const content = fs.readFileSync(srcPath, 'utf8');
      const newFileName = entry.name.replace('.js', '.esm.js');
      fs.writeFileSync(path.join(destDir, newFileName), content);
    }
  }
}
copyEsmFiles('dist-esm', 'dist');

// Clean temporary ESM directory
fs.rmSync('dist-esm', { recursive: true });

// Build TypeScript declarations
console.log('Building TypeScript declarations...');
execSync('npx tsc -p tsconfig.types.json', { stdio: 'inherit' });

console.log('Build complete!');
