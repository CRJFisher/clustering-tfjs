import '../tensorflow-helper';
import { findOptimalClusters, ClusterEvaluation } from '../../src/utils/findOptimalClusters';
import { make_random_stream } from '../../src/utils/rng';

function assertClusterEvaluationShape(evaluation: ClusterEvaluation, nSamples: number): void {
  // Required fields exist and have correct runtime type
  expect(typeof evaluation.k).toBe('number');
  expect(Number.isInteger(evaluation.k)).toBe(true);
  expect(evaluation.k).toBeGreaterThanOrEqual(2);

  expect(typeof evaluation.silhouette).toBe('number');
  expect(typeof evaluation.daviesBouldin).toBe('number');
  expect(typeof evaluation.calinskiHarabasz).toBe('number');
  expect(typeof evaluation.combinedScore).toBe('number');

  expect(Array.isArray(evaluation.labels)).toBe(true);
  expect(evaluation.labels).toHaveLength(nSamples);
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
    const result = await findOptimalClusters(data);

    expect(result).toHaveProperty('optimal');
    expect(result).toHaveProperty('evaluations');
    expect(Array.isArray(result.evaluations)).toBe(true);
    expect(result.evaluations.length).toBeGreaterThan(0);

    assertClusterEvaluationShape(result.optimal, data.length);

    for (const evaluation of result.evaluations) {
      assertClusterEvaluationShape(evaluation, data.length);
    }

    // wss is undefined for non-elbow method
    for (const evaluation of result.evaluations) {
      expect(evaluation.wss).toBeUndefined();
    }
  }, 30000);

  it('elbow method returns shape with wss field', async () => {
    const result = await findOptimalClusters(data, { method: 'elbow' });

    assertClusterEvaluationShape(result.optimal, data.length);

    for (const evaluation of result.evaluations) {
      assertClusterEvaluationShape(evaluation, data.length);
      expect(typeof evaluation.wss).toBe('number');
      expect(evaluation.wss!).toBeGreaterThanOrEqual(0);
    }
  }, 30000);

  it('silhouette method returns correct shape', async () => {
    const result = await findOptimalClusters(data, { method: 'silhouette' });

    assertClusterEvaluationShape(result.optimal, data.length);

    for (const evaluation of result.evaluations) {
      assertClusterEvaluationShape(evaluation, data.length);
      expect(evaluation.combinedScore).toBe(evaluation.silhouette);
    }
  }, 30000);

  it('evaluations are sorted by combinedScore descending', async () => {
    const result = await findOptimalClusters(data, {
      minClusters: 2,
      maxClusters: 4,
    });

    expect(result.evaluations.length).toBeGreaterThan(1);

    for (let i = 1; i < result.evaluations.length; i++) {
      expect(result.evaluations[i - 1].combinedScore)
        .toBeGreaterThanOrEqual(result.evaluations[i].combinedScore);
    }
  }, 30000);

  it('labels array length matches number of input samples', async () => {
    const sizes = [5, 10, 20];

    for (const size of sizes) {
      const rng = make_random_stream(42);
      const testData: number[][] = [];
      for (let i = 0; i < size; i++) {
        testData.push([rng.rand() * 10, rng.rand() * 10]);
      }

      const result = await findOptimalClusters(testData, {
        minClusters: 2,
        maxClusters: 3,
      });

      assertClusterEvaluationShape(result.optimal, size);

      for (const evaluation of result.evaluations) {
        assertClusterEvaluationShape(evaluation, size);
      }
    }
  }, 30000);

  it('evaluations length matches number of k values tested', async () => {
    const rng = make_random_stream(42);
    const largerData: number[][] = [];
    for (let i = 0; i < 15; i++) {
      largerData.push([rng.rand() * 10, rng.rand() * 10]);
    }

    const result = await findOptimalClusters(largerData, {
      minClusters: 2,
      maxClusters: 6,
    });

    expect(result.evaluations).toHaveLength(5);

    const kValues = new Set(result.evaluations.map(e => e.k));
    expect(kValues).toEqual(new Set([2, 3, 4, 5, 6]));

    for (const evaluation of result.evaluations) {
      assertClusterEvaluationShape(evaluation, largerData.length);
    }
  }, 30000);
});
