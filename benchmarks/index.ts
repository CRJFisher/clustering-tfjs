import * as tf from '@tensorflow/tfjs';
import { performance } from 'perf_hooks';
import { AgglomerativeClustering } from '../src/clustering/agglomerative';
import { SpectralClustering } from '../src/clustering/spectral';
import { KMeans } from '../src/clustering/kmeans';
import { SOM } from '../src/clustering/som';
import { HDBSCAN } from '../src/clustering/hdbscan';
import { make_blobs } from '../src/datasets/synthetic';

export interface BenchmarkResult {
  algorithm: string;
  backend: string;
  dataset_size: number;
  features: number;
  execution_time: number;
  memory_used: number;
  memory_peak: number;
  tensor_count: number;
  backend_init_time: number;
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
  // HDBSCAN front-half sweep (task-54). The front-half cost is O(n²·d), so
  // these vary d at a fixed n to isolate the dimensionality effect, then push n
  // toward the dense-matrix memory ceiling (5000) where the tfjs front-half is
  // expected to overtake the float64-JS pipeline. The n=10000 'large' config
  // above stays skipped for HDBSCAN by the O(n²) guard in run_benchmark_suite.
  { samples: 2000, features: 2, centers: 8, label: 'hdbscan_n2000_d2' },
  { samples: 2000, features: 16, centers: 8, label: 'hdbscan_n2000_d16' },
  { samples: 2000, features: 64, centers: 8, label: 'hdbscan_n2000_d64' },
  { samples: 2000, features: 128, centers: 8, label: 'hdbscan_n2000_d128' },
  { samples: 5000, features: 16, centers: 8, label: 'hdbscan_n5000_d16' },
  { samples: 5000, features: 128, centers: 8, label: 'hdbscan_n5000_d128' },
];

export async function benchmark_algorithm(
  algorithm:
    | 'kmeans'
    | 'spectral'
    | 'spectral_sparse'
    | 'agglomerative'
    | 'som'
    | 'hdbscan',
  config: BenchmarkConfig,
  backend: string,
): Promise<BenchmarkResult> {
  // Generate dataset
  const { X } = make_blobs({
    n_samples: config.samples,
    n_features: config.features,
    centers: config.centers,
    random_state: 42,
  });

  // Initialize backend
  const backend_init_start = performance.now();
  await tf.setBackend(backend);
  await tf.ready();
  const backend_init_time = performance.now() - backend_init_start;

  // Track memory
  const mem_before = tf.memory();

  // Run clustering
  const start = performance.now();
  let _labels: number[];

  switch (algorithm) {
    case 'kmeans': {
      const kmeans = new KMeans({ n_clusters: config.centers, random_state: 42 });
      await kmeans.fit(X);
      _labels = kmeans.labels_!;
      break;
    }
    case 'spectral': {
      const spectral = new SpectralClustering({
        n_clusters: config.centers,
        affinity: 'rbf',
        random_state: 42,
      });
      await spectral.fit(X);
      _labels = spectral.labels_!;
      break;
    }
    case 'spectral_sparse': {
      const spectral = new SpectralClustering({
        n_clusters: config.centers,
        affinity: 'nearest_neighbors',
        n_neighbors: Math.min(10, config.samples - 1),
        random_state: 42,
      });
      await spectral.fit(X);
      _labels = spectral.labels_!;
      break;
    }
    case 'agglomerative': {
      const agglo = new AgglomerativeClustering({
        n_clusters: config.centers,
        linkage: 'average',
      });
      await agglo.fit(X);
      _labels = agglo.labels_!;
      break;
    }
    case 'som': {
      // For SOM, use a square grid approximately matching the number of clusters
      const grid_size = Math.ceil(Math.sqrt(config.centers));
      const som = new SOM({
        grid_width: grid_size,
        grid_height: grid_size,
        topology: 'rectangular',
        initialization: 'pca',
        num_epochs: 50,  // Reduced for benchmarking
        random_state: 42,
      });
      await som.fit(X);
      _labels = som.labels_!;
      break;
    }
    case 'hdbscan': {
      // Density-based: no preset cluster count; min_cluster_size scales mildly
      // with the dataset so blobs resolve as clusters rather than noise.
      const hdbscan = new HDBSCAN({
        min_cluster_size: Math.max(5, Math.floor(config.samples / 50)),
      });
      await hdbscan.fit(X);
      _labels = hdbscan.labels_!;
      break;
    }
  }

  const execution_time = performance.now() - start;
  const mem_after = tf.memory();

  return {
    algorithm,
    backend,
    dataset_size: config.samples,
    features: config.features,
    execution_time,
    memory_used: mem_after.numBytes - mem_before.numBytes,
    memory_peak: mem_after.numBytes,
    tensor_count: mem_after.numTensors - mem_before.numTensors,
    backend_init_time,
  };
}

export async function get_available_backends(): Promise<string[]> {
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

export async function run_benchmark_suite(): Promise<BenchmarkResult[]> {
  const results: BenchmarkResult[] = [];
  const backends = await get_available_backends();
  const algorithms: Array<
    | 'kmeans'
    | 'spectral'
    | 'spectral_sparse'
    | 'agglomerative'
    | 'som'
    | 'hdbscan'
  > = [
    'kmeans',
    'spectral',
    'spectral_sparse',
    'agglomerative',
    'som',
    'hdbscan',
  ];

  console.log(`Available backends: ${backends.join(', ')}`);

  for (const backend of backends) {
    for (const algorithm of algorithms) {
      for (const config of BENCHMARK_CONFIGS) {
        // HDBSCAN builds dense O(n²) distance / mutual-reachability matrices, so
        // it is benchmarked only up to its documented ~5k-sample ceiling; larger
        // datasets are skipped (and logged) rather than OOM the run.
        if (algorithm === 'hdbscan' && config.samples > 5000) {
          console.log(
            `Skipping hdbscan on ${config.label} dataset (n=${config.samples} exceeds the dense O(n²) ceiling).`,
          );
          continue;
        }

        console.log(
          `Benchmarking ${algorithm} on ${backend} with ${config.label} dataset...`,
        );

        try {
          const result = await benchmark_algorithm(algorithm, config, backend);
          results.push(result);

          console.log(`  Time: ${result.execution_time.toFixed(2)}ms`);
          console.log(
            `  Memory: ${(result.memory_used / 1024 / 1024).toFixed(2)}MB`,
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

export function format_benchmark_results(results: BenchmarkResult[]): string {
  let output = '# Benchmark Results\n\n';
  output +=
    '| Algorithm | Backend | Dataset | Time (ms) | Memory (MB) | Backend Init (ms) |\n';
  output +=
    '|-----------|---------|---------|-----------|-------------|-------------------|\n';

  for (const result of results) {
    const dataset = `${result.dataset_size}x${result.features}`;
    const time = result.execution_time.toFixed(2);
    const memory = (result.memory_used / 1024 / 1024).toFixed(2);
    const init = result.backend_init_time.toFixed(2);

    output += `| ${result.algorithm} | ${result.backend} | ${dataset} | ${time} | ${memory} | ${init} |\n`;
  }

  return output;
}
