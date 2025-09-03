import * as tf from '@tensorflow/tfjs';
import { performance } from 'perf_hooks';
import { AgglomerativeClustering } from '../clustering/agglomerative';
import { SpectralClustering } from '../clustering/spectral';
import { KMeans } from '../clustering/kmeans';
import { SOM } from '../clustering/som';
import { makeBlobs } from '../datasets/synthetic';

export interface BenchmarkResult {
  algorithm: string;
  backend: string;
  datasetSize: number;
  features: number;
  executionTime: number;
  memoryUsed: number;
  memoryPeak: number;
  tensorCount: number;
  backendInitTime: number;
  accuracy?: number;
}

export interface BenchmarkConfig {
  samples: number;
  features: number;
  centers: number;
  label: string;
}

export const BENCHMARK_CONFIGS: BenchmarkConfig[] = [
  { samples: 100, features: 10, centers: 3, label: 'small' },
  { samples: 1000, features: 50, centers: 5, label: 'medium' },
  { samples: 10000, features: 100, centers: 10, label: 'large' },
];

export async function benchmarkAlgorithm(
  algorithm: 'kmeans' | 'spectral' | 'agglomerative' | 'som',
  config: BenchmarkConfig,
  backend: string,
): Promise<BenchmarkResult> {
  // Generate dataset
  const { X } = makeBlobs({
    nSamples: config.samples,
    nFeatures: config.features,
    centers: config.centers,
    randomState: 42,
  });

  // Initialize backend
  const backendInitStart = performance.now();
  await tf.setBackend(backend);
  await tf.ready();
  const backendInitTime = performance.now() - backendInitStart;

  // Track memory
  const memBefore = tf.memory();

  // Run clustering
  const start = performance.now();
  let _labels: number[];

  switch (algorithm) {
    case 'kmeans': {
      const kmeans = new KMeans({ nClusters: config.centers, randomState: 42 });
      await kmeans.fit(X);
      _labels = Array.isArray(kmeans.labels_)
        ? kmeans.labels_
        : ((await kmeans.labels_!.array()) as number[]);
      break;
    }
    case 'spectral': {
      const spectral = new SpectralClustering({
        nClusters: config.centers,
        affinity: 'rbf',
        randomState: 42,
      });
      await spectral.fit(X);
      _labels = spectral.labels_!;
      break;
    }
    case 'agglomerative': {
      const agglo = new AgglomerativeClustering({
        nClusters: config.centers,
        linkage: 'average',
      });
      await agglo.fit(X);
      _labels = Array.isArray(agglo.labels_)
        ? agglo.labels_
        : ((await agglo.labels_!.array()) as number[]);
      break;
    }
    case 'som': {
      // For SOM, use a square grid approximately matching the number of clusters
      const gridSize = Math.ceil(Math.sqrt(config.centers));
      const som = new SOM({
        nClusters: gridSize * gridSize,
        gridWidth: gridSize,
        gridHeight: gridSize,
        topology: 'rectangular',
        initialization: 'pca',
        numEpochs: 50,  // Reduced for benchmarking
        randomState: 42,
      });
      await som.fit(X);
      _labels = Array.isArray(som.labels_)
        ? som.labels_
        : ((await som.labels_!.array()) as number[]);
      break;
    }
  }

  const executionTime = performance.now() - start;
  const memAfter = tf.memory();

  return {
    algorithm,
    backend,
    datasetSize: config.samples,
    features: config.features,
    executionTime,
    memoryUsed: memAfter.numBytes - memBefore.numBytes,
    memoryPeak: memAfter.numBytes,
    tensorCount: memAfter.numTensors - memBefore.numTensors,
    backendInitTime,
  };
}

export async function getAvailableBackends(): Promise<string[]> {
  const backends: string[] = ['cpu'];

  // TODO: Add WASM backend check when types are available
  // Currently commented out to avoid TypeScript errors
  // try {
  //   await import('@tensorflow/tfjs-backend-wasm');
  //   backends.push('wasm');
  // } catch {}

  // Check if tfjs-node is available
  try {
    await import('@tensorflow/tfjs-node');
    backends.push('tensorflow');
  } catch {
    // tfjs-node not available, skip
  }

  // Check if tfjs-node-gpu is available
  try {
    // @ts-expect-error - tfjs-node-gpu may not be installed, this is expected
    await import('@tensorflow/tfjs-node-gpu');
    backends.push('tensorflow-gpu');
  } catch {
    // tfjs-node-gpu not available, skip
  }

  return backends;
}

export async function runBenchmarkSuite(): Promise<BenchmarkResult[]> {
  const results: BenchmarkResult[] = [];
  const backends = await getAvailableBackends();
  const algorithms: Array<'kmeans' | 'spectral' | 'agglomerative' | 'som'> = [
    'kmeans',
    'spectral',
    'agglomerative',
    'som',
  ];

  console.log(`Available backends: ${backends.join(', ')}`);

  for (const backend of backends) {
    for (const algorithm of algorithms) {
      for (const config of BENCHMARK_CONFIGS) {
        console.log(
          `Benchmarking ${algorithm} on ${backend} with ${config.label} dataset...`,
        );

        try {
          const result = await benchmarkAlgorithm(algorithm, config, backend);
          results.push(result);

          console.log(`  Time: ${result.executionTime.toFixed(2)}ms`);
          console.log(
            `  Memory: ${(result.memoryUsed / 1024 / 1024).toFixed(2)}MB`,
          );
        } catch (error) {
          console.error(
            `  Failed: ${error instanceof Error ? error.message : String(error)}`,
          );
        }
      }
    }
  }

  return results;
}

export function formatBenchmarkResults(results: BenchmarkResult[]): string {
  let output = '# Benchmark Results\n\n';
  output +=
    '| Algorithm | Backend | Dataset | Time (ms) | Memory (MB) | Backend Init (ms) |\n';
  output +=
    '|-----------|---------|---------|-----------|-------------|-------------------|\n';

  for (const result of results) {
    const dataset = `${result.datasetSize}x${result.features}`;
    const time = result.executionTime.toFixed(2);
    const memory = (result.memoryUsed / 1024 / 1024).toFixed(2);
    const init = result.backendInitTime.toFixed(2);

    output += `| ${result.algorithm} | ${result.backend} | ${dataset} | ${time} | ${memory} | ${init} |\n`;
  }

  return output;
}
