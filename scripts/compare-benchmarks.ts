#!/usr/bin/env node
import { readFileSync, readdirSync, writeFileSync } from 'fs';
import { join } from 'path';
import { BenchmarkResult } from '../benchmarks';
import { analyze_backend_performance, generate_backend_recommendations } from '../benchmarks/compare';
import * as yaml from 'js-yaml';

function load_latest_benchmark(): BenchmarkResult[] {
  const benchmark_dir = join(process.cwd(), 'benchmarks');
  const files = readdirSync(benchmark_dir)
    .filter(f => f.endsWith('.yaml') || f.endsWith('.json'))
    .sort()
    .reverse();
  
  if (files.length === 0) {
    throw new Error('No benchmark results found. Run npm run benchmark first.');
  }
  
  const latest_file = files[0];
  console.log(`Loading benchmark results from: ${latest_file}\n`);
  
  const content = readFileSync(join(benchmark_dir, latest_file), 'utf8');
  if (latest_file.endsWith('.yaml')) {
    return yaml.load(content) as BenchmarkResult[];
  } else {
    return JSON.parse(content);
  }
}

async function main() {
  try {
    const results = load_latest_benchmark();
    const comparisons = analyze_backend_performance(results);
    const recommendations = generate_backend_recommendations(comparisons);
    
    console.log(recommendations);
    
    // Save recommendations
    const output_path = join(process.cwd(), 'benchmarks', 'backend-recommendations.md');
    writeFileSync(output_path, recommendations);
    console.log(`\nRecommendations saved to: ${output_path}`);
    
  } catch (error) {
    console.error('Error:', error instanceof Error ? error.message : String(error));
    process.exit(1);
  }
}

main();