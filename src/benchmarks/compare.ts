import { BenchmarkResult } from './index';

export interface BackendComparison {
  backend: string;
  algorithm: string;
  datasetSize: string;
  speedupVsCPU: number;
  memoryRatio: number;
  recommendation: string;
}

export function analyzeBackendPerformance(results: BenchmarkResult[]): BackendComparison[] {
  const comparisons: BackendComparison[] = [];
  
  // Group results by algorithm and dataset size
  const grouped = new Map<string, BenchmarkResult[]>();
  
  for (const result of results) {
    const key = `${result.algorithm}-${result.datasetSize}x${result.features}`;
    if (!grouped.has(key)) {
      grouped.set(key, []);
    }
    grouped.get(key)!.push(result);
  }
  
  // Compare each backend against CPU baseline
  for (const [_key, groupResults] of grouped) {
    const cpuResult = groupResults.find(r => r.backend === 'cpu');
    if (!cpuResult) continue;
    
    for (const result of groupResults) {
      const speedup = cpuResult.executionTime / result.executionTime;
      const memoryRatio = result.memoryUsed / cpuResult.memoryUsed;
      
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
      if (memoryRatio > 2) {
        recommendation += ' (high memory usage)';
      }
      
      comparisons.push({
        backend: result.backend,
        algorithm: result.algorithm,
        datasetSize: `${result.datasetSize}x${result.features}`,
        speedupVsCPU: speedup,
        memoryRatio,
        recommendation,
      });
    }
  }
  
  return comparisons;
}

export function generateBackendRecommendations(comparisons: BackendComparison[]): string {
  let output = '# Backend Recommendations\n\n';
  
  // Group by dataset size
  const bySize = new Map<string, BackendComparison[]>();
  for (const comp of comparisons) {
    const size = comp.datasetSize;
    if (!bySize.has(size)) {
      bySize.set(size, []);
    }
    bySize.get(size)!.push(comp);
  }
  
  // Sort by dataset size
  const sizes = Array.from(bySize.keys()).sort((a, b) => {
    const aNum = parseInt(a.split('x')[0]);
    const bNum = parseInt(b.split('x')[0]);
    return aNum - bNum;
  });
  
  for (const size of sizes) {
    output += `## Dataset Size: ${size}\n\n`;
    output += '| Algorithm | Backend | Speedup vs CPU | Memory Ratio | Recommendation |\n';
    output += '|-----------|---------|----------------|--------------|----------------|\n';
    
    const sizeComps = bySize.get(size)!;
    // Sort by algorithm then speedup
    sizeComps.sort((a, b) => {
      if (a.algorithm !== b.algorithm) {
        return a.algorithm.localeCompare(b.algorithm);
      }
      return b.speedupVsCPU - a.speedupVsCPU;
    });
    
    for (const comp of sizeComps) {
      if (comp.backend === 'cpu') continue; // Skip CPU baseline
      
      output += `| ${comp.algorithm} | ${comp.backend} | ${comp.speedupVsCPU.toFixed(2)}x | ${comp.memoryRatio.toFixed(2)}x | ${comp.recommendation} |\n`;
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