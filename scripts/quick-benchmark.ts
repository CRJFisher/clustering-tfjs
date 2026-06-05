#!/usr/bin/env node
import { benchmark_algorithm, format_benchmark_results, get_available_backends } from '../benchmarks';
import { analyze_backend_performance, generate_backend_recommendations } from '../benchmarks/compare';

// Quick benchmark with smaller datasets
const QUICK_CONFIGS = [
  { samples: 100, features: 10, centers: 3, label: 'small' },
  { samples: 500, features: 20, centers: 5, label: 'medium' },
];

async function main() {
  console.log('Running quick benchmark...\n');
  
  const backends = await get_available_backends();
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
          const result = await benchmark_algorithm(algorithm, config, backend);
          results.push(result);
          
          console.log(`    Time: ${result.execution_time.toFixed(2)}ms`);
          console.log(`    Memory: ${(result.memory_used / 1024 / 1024).toFixed(2)}MB`);
        } catch (error) {
          console.error(`    Failed: ${error instanceof Error ? error.message : String(error)}`);
        }
      }
    }
  }
  
  console.log('\n' + format_benchmark_results(results));
  
  // Generate backend comparison
  const comparisons = analyze_backend_performance(results);
  const recommendations = generate_backend_recommendations(comparisons);
  console.log('\n' + recommendations);
}

main().catch(console.error);