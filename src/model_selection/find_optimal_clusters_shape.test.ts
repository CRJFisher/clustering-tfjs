import '../../test_support/tensorflow_helper';
import { find_optimal_clusters, ClusterEvaluation } from './find_optimal_clusters';
import { make_random_stream } from '../random';

function assert_cluster_evaluation_shape(evaluation: ClusterEvaluation, n_samples: number): void {
  // Required fields exist and have correct runtime type
  expect(typeof evaluation.k).toBe('number');
  expect(Number.isInteger(evaluation.k)).toBe(true);
  expect(evaluation.k).toBeGreaterThanOrEqual(2);

  expect(typeof evaluation.silhouette).toBe('number');
  expect(typeof evaluation.davies_bouldin).toBe('number');
  expect(typeof evaluation.calinski_harabasz).toBe('number');
  expect(typeof evaluation.combined_score).toBe('number');

  expect(Array.isArray(evaluation.labels)).toBe(true);
  expect(evaluation.labels).toHaveLength(n_samples);
  for (const label of evaluation.labels) {
    expect(typeof label).toBe('number');
    expect(Number.isInteger(label)).toBe(true);
    expect(label).toBeGreaterThanOrEqual(0);
    expect(label).toBeLessThan(evaluation.k);
  }
}

describe('findOptimalClusters return shape', () => {
  const data = [
    [1, 2], [1.5, 1.8], [5, 8], [8, 8],
    [1, 0.6], [9, 11], [1.2, 1.9], [8.5, 7.5],
  ];

  it('default method (combined) returns correct shape', async () => {
    const result = await find_optimal_clusters(data);

    expect(result).toHaveProperty('optimal');
    expect(result).toHaveProperty('evaluations');
    expect(Array.isArray(result.evaluations)).toBe(true);
    expect(result.evaluations.length).toBeGreaterThan(0);

    assert_cluster_evaluation_shape(result.optimal, data.length);

    for (const evaluation of result.evaluations) {
      assert_cluster_evaluation_shape(evaluation, data.length);
    }

    // wss is undefined for non-elbow method
    for (const evaluation of result.evaluations) {
      expect(evaluation.wss).toBeUndefined();
    }
  }, 30000);

  it('elbow method returns shape with wss field', async () => {
    const result = await find_optimal_clusters(data, { method: 'elbow' });

    assert_cluster_evaluation_shape(result.optimal, data.length);

    for (const evaluation of result.evaluations) {
      assert_cluster_evaluation_shape(evaluation, data.length);
      expect(typeof evaluation.wss).toBe('number');
      expect(evaluation.wss!).toBeGreaterThanOrEqual(0);
    }
  }, 30000);

  it('silhouette method returns correct shape', async () => {
    const result = await find_optimal_clusters(data, { method: 'silhouette' });

    assert_cluster_evaluation_shape(result.optimal, data.length);

    for (const evaluation of result.evaluations) {
      assert_cluster_evaluation_shape(evaluation, data.length);
      expect(evaluation.combined_score).toBe(evaluation.silhouette);
    }
  }, 30000);

  it('evaluations are sorted by combinedScore descending', async () => {
    const result = await find_optimal_clusters(data, {
      min_clusters: 2,
      max_clusters: 4,
    });

    expect(result.evaluations.length).toBeGreaterThan(1);

    for (let i = 1; i < result.evaluations.length; i++) {
      expect(result.evaluations[i - 1].combined_score)
        .toBeGreaterThanOrEqual(result.evaluations[i].combined_score);
    }
  }, 30000);

  it('labels array length matches number of input samples', async () => {
    const sizes = [5, 10, 20];

    for (const size of sizes) {
      const rng = make_random_stream(42);
      const test_data: number[][] = [];
      for (let i = 0; i < size; i++) {
        test_data.push([rng.rand() * 10, rng.rand() * 10]);
      }

      const result = await find_optimal_clusters(test_data, {
        min_clusters: 2,
        max_clusters: 3,
      });

      assert_cluster_evaluation_shape(result.optimal, size);

      for (const evaluation of result.evaluations) {
        assert_cluster_evaluation_shape(evaluation, size);
      }
    }
  }, 30000);

  it('evaluations length matches number of k values tested', async () => {
    const rng = make_random_stream(42);
    const larger_data: number[][] = [];
    for (let i = 0; i < 15; i++) {
      larger_data.push([rng.rand() * 10, rng.rand() * 10]);
    }

    const result = await find_optimal_clusters(larger_data, {
      min_clusters: 2,
      max_clusters: 6,
    });

    expect(result.evaluations).toHaveLength(5);

    const k_values = new Set(result.evaluations.map(e => e.k));
    expect(k_values).toEqual(new Set([2, 3, 4, 5, 6]));

    for (const evaluation of result.evaluations) {
      assert_cluster_evaluation_shape(evaluation, larger_data.length);
    }
  }, 30000);
});
