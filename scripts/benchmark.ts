#!/usr/bin/env node
import { runBenchmarkSuite, formatBenchmarkResults } from '../src/benchmarks';
import { writeFileSync } from 'fs';
import { join } from 'path';

async function main() {
  console.log('Starting benchmark suite...\n');
  
  const startTime = Date.now();
  const results = await runBenchmarkSuite();
  const duration = (Date.now() - startTime) / 1000;
  
  console.log(`\nBenchmark completed in ${duration.toFixed(1)}s\n`);
  
  // Print results to console
  const formatted = formatBenchmarkResults(results);
  console.log(formatted);
  
  // Save results to file with timestamp
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const outputPath = join(process.cwd(), 'benchmarks', `results-${timestamp}.json`);
  
  // Create benchmarks directory if it doesn't exist
  const fs = await import('fs');
  if (!fs.existsSync('benchmarks')) {
    fs.mkdirSync('benchmarks');
  }
  
  // Save JSON results
  writeFileSync(outputPath, JSON.stringify(results, null, 2));
  console.log(`\nResults saved to: ${outputPath}`);
  
  // Save markdown report
  const mdPath = outputPath.replace('.json', '.md');
  writeFileSync(mdPath, formatted);
  console.log(`Report saved to: ${mdPath}`);
}

main().catch(console.error);