import { BenchmarkResult } from './';

export interface BackendComparison {
  backend: string;
  algorithm: string;
  dataset_size: string;
  speedup_vs_cpu: number;
  memory_ratio: number;
  recommendation: string;
}

export function analyze_backend_performance(
  results: BenchmarkResult[],
): BackendComparison[] {
  const comparisons: BackendComparison[] = [];

  // Group results by algorithm and dataset size
  const grouped = new Map<string, BenchmarkResult[]>();

  for (const result of results) {
    const key = `${result.algorithm}-${result.dataset_size}x${result.features}`;
    if (!grouped.has(key)) {
      grouped.set(key, []);
    }
    grouped.get(key)!.push(result);
  }

  // Compare each backend against CPU baseline
  for (const [_key, group_results] of grouped) {
    const cpu_result = group_results.find((r) => r.backend === 'cpu');
    if (!cpu_result) continue;

    for (const result of group_results) {
      const speedup = cpu_result.execution_time / result.execution_time;
      const memory_ratio = result.memory_used / cpu_result.memory_used;

      let recommendation = '';
      if (speedup > 10) {
        recommendation = 'Highly recommended';
      } else if (speedup > 2) {
        recommendation = 'Recommended';
      } else if (speedup > 1.2) {
        recommendation = 'Minor improvement';
      } else if (speedup < 0.8) {
        recommendation = 'Not recommended (slower)';
      } else {
        recommendation = 'No significant benefit';
      }

      // Adjust recommendation based on memory usage
      if (memory_ratio > 2) {
        recommendation += ' (high memory usage)';
      }

      comparisons.push({
        backend: result.backend,
        algorithm: result.algorithm,
        dataset_size: `${result.dataset_size}x${result.features}`,
        speedup_vs_cpu: speedup,
        memory_ratio,
        recommendation,
      });
    }
  }

  return comparisons;
}

export function generate_backend_recommendations(
  comparisons: BackendComparison[],
): string {
  let output = '# Backend Recommendations\n\n';

  // Group by dataset size
  const by_size = new Map<string, BackendComparison[]>();
  for (const comp of comparisons) {
    const size = comp.dataset_size;
    if (!by_size.has(size)) {
      by_size.set(size, []);
    }
    by_size.get(size)!.push(comp);
  }

  // Sort by dataset size
  const sizes = Array.from(by_size.keys()).sort((a, b) => {
    const a_num = parseInt(a.split('x')[0]);
    const b_num = parseInt(b.split('x')[0]);
    return a_num - b_num;
  });

  for (const size of sizes) {
    output += `## Dataset Size: ${size}\n\n`;
    output +=
      '| Algorithm | Backend | Speedup vs CPU | Memory Ratio | Recommendation |\n';
    output +=
      '|-----------|---------|----------------|--------------|----------------|\n';

    const size_comps = by_size.get(size)!;
    // Sort by algorithm then speedup
    size_comps.sort((a, b) => {
      if (a.algorithm !== b.algorithm) {
        return a.algorithm.localeCompare(b.algorithm);
      }
      return b.speedup_vs_cpu - a.speedup_vs_cpu;
    });

    for (const comp of size_comps) {
      if (comp.backend === 'cpu') continue; // Skip CPU baseline

      output += `| ${comp.algorithm} | ${comp.backend} | ${comp.speedup_vs_cpu.toFixed(2)}x | ${comp.memory_ratio.toFixed(2)}x | ${comp.recommendation} |\n`;
    }
    output += '\n';
  }

  // Add summary recommendations
  output += '## Summary Recommendations\n\n';
  output += '### Small Datasets (<1k samples)\n';
  output += '- **Recommended**: WASM backend (2-3x speedup, low overhead)\n';
  output += '- **Alternative**: CPU backend (simplest, no dependencies)\n\n';

  output += '### Medium Datasets (1k-10k samples)\n';
  output += '- **Recommended**: tfjs-node (5-20x speedup)\n';
  output += '- **Alternative**: WASM (if native dependencies problematic)\n\n';

  output += '### Large Datasets (>10k samples)\n';
  output += '- **Recommended**: tfjs-node-gpu (if CUDA available)\n';
  output += '- **Alternative**: tfjs-node (still significant speedup)\n\n';

  output += '### Browser Environment\n';
  output += '- **Recommended**: WebGL backend (GPU acceleration)\n';
  output += '- **Fallback**: WASM backend\n';

  return output;
}
