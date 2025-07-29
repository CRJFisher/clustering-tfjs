import { benchmarkAlgorithm, getAvailableBackends, BENCHMARK_CONFIGS } from '../../src/benchmarks';

describe('Benchmarking System', () => {
  it('should detect available backends', async () => {
    const backends = await getAvailableBackends();
    expect(backends).toContain('cpu');
    // WASM backend check is currently disabled due to missing types
    // expect(backends).toContain('wasm');
    // tfjs-node might not be available in all environments
  });

  it('should run a simple benchmark', async () => {
    const config = BENCHMARK_CONFIGS[0]; // small dataset
    const result = await benchmarkAlgorithm('kmeans', config, 'cpu');
    
    expect(result.algorithm).toBe('kmeans');
    expect(result.backend).toBe('cpu');
    expect(result.datasetSize).toBe(config.samples);
    expect(result.executionTime).toBeGreaterThan(0);
    expect(result.memoryUsed).toBeGreaterThanOrEqual(0);
    expect(result.tensorCount).toBeGreaterThanOrEqual(0);
  });

  it('should benchmark all algorithms', async () => {
    const config = BENCHMARK_CONFIGS[0]; // small dataset
    const algorithms: Array<'kmeans' | 'spectral' | 'agglomerative'> = 
      ['kmeans', 'spectral', 'agglomerative'];
    
    for (const algo of algorithms) {
      const result = await benchmarkAlgorithm(algo, config, 'cpu');
      expect(result.algorithm).toBe(algo);
      expect(result.executionTime).toBeGreaterThan(0);
    }
  });
});