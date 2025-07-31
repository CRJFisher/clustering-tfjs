import puppeteer from 'puppeteer';
import { createServer } from 'http';

export interface BrowserBenchmarkResult {
  algorithm: string;
  backend: 'webgl' | 'wasm' | 'cpu';
  datasetSize: number;
  features: number;
  executionTime: number;
  memoryUsed: number;
}

const benchmarkHTML = `
<!DOCTYPE html>
<html>
<head>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
</head>
<body>
  <script>
    window.runBenchmark = async function(config) {
      const { algorithm, backend, samples, features, centers } = config;
      
      // Set backend
      await tf.setBackend(backend);
      await tf.ready();
      
      // Generate random data (simplified makeBlobs)
      const X = tf.randomNormal([samples, features]);
      
      // Track memory
      const memBefore = tf.memory();
      const start = performance.now();
      
      // Run clustering (simplified - would need actual implementations)
      // For now, just do some tensor operations to simulate work
      let result;
      if (algorithm === 'kmeans') {
        // Simulate k-means operations
        const centroids = tf.randomNormal([centers, features]);
        for (let i = 0; i < 10; i++) {
          const distances = tf.sum(tf.square(tf.sub(
            tf.expandDims(X, 1),
            tf.expandDims(centroids, 0)
          )), 2);
          const labels = tf.argMin(distances, 1);
          labels.dispose();
          distances.dispose();
        }
        centroids.dispose();
      }
      
      const executionTime = performance.now() - start;
      const memAfter = tf.memory();
      
      X.dispose();
      
      return {
        algorithm,
        backend,
        datasetSize: samples,
        features,
        executionTime,
        memoryUsed: memAfter.numBytes - memBefore.numBytes,
      };
    };
  </script>
</body>
</html>
`;

export async function benchmarkInBrowser(
  configs: Array<{
    algorithm: string;
    backend: 'webgl' | 'wasm' | 'cpu';
    samples: number;
    features: number;
    centers: number;
  }>,
): Promise<BrowserBenchmarkResult[]> {
  // Create a simple HTTP server to serve the benchmark page
  const server = createServer((req, res) => {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(benchmarkHTML);
  });

  await new Promise<void>((resolve) => {
    server.listen(0, () => resolve());
  });

  const address = server.address();
  if (!address || typeof address !== 'object') {
    throw new Error('Failed to get server address');
  }
  const port = address.port;

  // Launch headless browser
  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  const page = await browser.newPage();
  await page.goto(`http://localhost:${port}`);

  const results: BrowserBenchmarkResult[] = [];

  for (const config of configs) {
    console.log(
      `Running ${config.algorithm} on ${config.backend} in browser...`,
    );

    try {
      const result = await page.evaluate(
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (cfg: any) => (window as any).runBenchmark(cfg),
        config,
      );
      results.push(result);
    } catch (error: unknown) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      console.error(`Failed: ${errorMessage}`);
    }
  }

  await browser.close();
  server.close();

  return results;
}
