#!/usr/bin/env node
import { benchmarkAlgorithm, formatBenchmarkResults, getAvailableBackends } from '../src/benchmarks';
import { analyzeBackendPerformance, generateBackendRecommendations } from '../src/benchmarks/compare';

// Quick benchmark with smaller datasets
const QUICK_CONFIGS = [
  { samples: 100, features: 10, centers: 3, label: 'small' },
  { samples: 500, features: 20, centers: 5, label: 'medium' },
];

async function main() {
  console.log('Running quick benchmark...\n');
  
  const backends = await getAvailableBackends();
  console.log(`Available backends: ${backends.join(', ')}\n`);
  
  const results = [];
  
  // Only test kmeans and agglomerative (spectral is too slow on CPU)
  const algorithms: Array<'kmeans' | 'agglomerative'> = ['kmeans', 'agglomerative'];
  
  for (const backend of backends) {
    console.log(`\nTesting ${backend} backend:`);
    
    for (const algorithm of algorithms) {
      for (const config of QUICK_CONFIGS) {
        console.log(`  ${algorithm} on ${config.label} dataset...`);
        
        try {
          const result = await benchmarkAlgorithm(algorithm, config, backend);
          results.push(result);
          
          console.log(`    Time: ${result.executionTime.toFixed(2)}ms`);
          console.log(`    Memory: ${(result.memoryUsed / 1024 / 1024).toFixed(2)}MB`);
        } catch (error) {
          console.error(`    Failed: ${error instanceof Error ? error.message : String(error)}`);
        }
      }
    }
  }
  
  console.log('\n' + formatBenchmarkResults(results));
  
  // Generate backend comparison
  const comparisons = analyzeBackendPerformance(results);
  const recommendations = generateBackendRecommendations(comparisons);
  console.log('\n' + recommendations);
}

main().catch(console.error);