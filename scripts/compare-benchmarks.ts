#!/usr/bin/env node
import { readFileSync, readdirSync, writeFileSync } from 'fs';
import { join } from 'path';
import { BenchmarkResult } from '../src/benchmarks';
import { analyzeBackendPerformance, generateBackendRecommendations } from '../src/benchmarks/compare';
import * as yaml from 'js-yaml';

function loadLatestBenchmark(): BenchmarkResult[] {
  const benchmarkDir = join(process.cwd(), 'benchmarks');
  const files = readdirSync(benchmarkDir)
    .filter(f => f.endsWith('.yaml') || f.endsWith('.json'))
    .sort()
    .reverse();
  
  if (files.length === 0) {
    throw new Error('No benchmark results found. Run npm run benchmark first.');
  }
  
  const latestFile = files[0];
  console.log(`Loading benchmark results from: ${latestFile}\n`);
  
  const content = readFileSync(join(benchmarkDir, latestFile), 'utf8');
  if (latestFile.endsWith('.yaml')) {
    return yaml.load(content) as BenchmarkResult[];
  } else {
    return JSON.parse(content);
  }
}

async function main() {
  try {
    const results = loadLatestBenchmark();
    const comparisons = analyzeBackendPerformance(results);
    const recommendations = generateBackendRecommendations(comparisons);
    
    console.log(recommendations);
    
    // Save recommendations
    const outputPath = join(process.cwd(), 'benchmarks', 'backend-recommendations.md');
    writeFileSync(outputPath, recommendations);
    console.log(`\nRecommendations saved to: ${outputPath}`);
    
  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  }
}

main();