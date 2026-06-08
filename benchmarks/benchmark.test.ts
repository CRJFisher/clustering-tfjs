import { benchmark_algorithm, get_available_backends, BENCHMARK_CONFIGS } from './';

describe('Benchmarking System', () => {
  it('should detect available backends', async () => {
    const backends = await get_available_backends();
    expect(backends).toContain('cpu');
    // WASM backend check is currently disabled due to missing types
    // expect(backends).to_contain('wasm');
    // tfjs-node might not be available in all environments
  });

  it('should run a simple benchmark', async () => {
    const config = BENCHMARK_CONFIGS[0]; // small dataset
    const result = await benchmark_algorithm('kmeans', config, 'cpu');
    
    expect(result.algorithm).toBe('kmeans');
    expect(result.backend).toBe('cpu');
    expect(result.dataset_size).toBe(config.samples);
    expect(result.execution_time).toBeGreaterThan(0);
    expect(result.memory_used).toBeGreaterThanOrEqual(0);
    expect(result.tensor_count).toBeGreaterThanOrEqual(0);
  });

  it('should benchmark all algorithms', async () => {
    const config = BENCHMARK_CONFIGS[0]; // small dataset
    const algorithms: Array<
      'kmeans' | 'spectral' | 'spectral_sparse' | 'agglomerative' | 'som'
    > = ['kmeans', 'spectral', 'spectral_sparse', 'agglomerative', 'som'];
    
    for (const algo of algorithms) {
      const result = await benchmark_algorithm(algo, config, 'cpu');
      expect(result.algorithm).toBe(algo);
      expect(result.execution_time).toBeGreaterThan(0);
    }
  });
});