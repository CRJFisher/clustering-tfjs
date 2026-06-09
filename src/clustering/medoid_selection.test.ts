import fs from 'fs';
import path from 'path';

import { select_medoids } from './medoid_selection';
import { AgglomerativeClustering, SpectralClustering } from '..';
import type { ClusteringMetric } from './types';
import * as tf from '../../test_support/tensorflow_helper';

const FIXTURE_DIR = path.join(process.cwd(), '__fixtures__', 'agglomerative');

interface MedoidFixture {
  X: number[][];
  params: { n_clusters: number; linkage: string; metric: ClusteringMetric };
  labels: number[];
  medoid_indices: number[];
  member_indices: number[][];
  distances_to_mean: number[][];
}

describe('select_medoids', () => {
  it('returns -1 for empty clusters and maps to original indices', async () => {
    const X = [
      [0, 0],
      [0.1, 0],
      [10, 10],
    ];
    const labels = [0, 0, 2]; // cluster 1 is empty
    const { indices, distances } = await select_medoids(X, labels, 3, 'euclidean');
    expect(indices.length).toBe(3);
    expect(indices[1]).toBe(-1);
    expect(distances[1]).toBe(Number.POSITIVE_INFINITY);
    // Cluster 0 medoid is one of {0,1}; cluster 2 medoid is index 2.
    expect([0, 1]).toContain(indices[0]);
    expect(indices[2]).toBe(2);
  });

  it('ignores noise (-1) labels', async () => {
    const X = [
      [0, 0],
      [0.1, 0],
      [99, 99],
    ];
    const { indices } = await select_medoids(X, [0, 0, -1], 1, 'euclidean');
    expect([0, 1]).toContain(indices[0]);
  });

  it('returns empty slots for empty input', async () => {
    const { indices, distances } = await select_medoids([], [], 2, 'euclidean');
    expect(Array.from(indices)).toEqual([-1, -1]);
    expect(Array.from(distances)).toEqual([
      Number.POSITIVE_INFINITY,
      Number.POSITIVE_INFINITY,
    ]);
  });

  it('selects the manhattan medoid, which differs from the euclidean one', async () => {
    // Cluster mean is (0, 0). Distances to it:
    //   point 0 (1.4, 0):    L2 = 1.40, L1 = 1.4
    //   point 1 (0.8, 0.8):  L2 = 1.13, L1 = 1.6
    //   point 2 (-2.2,-0.8): L2 = 2.34, L1 = 3.0
    // -> euclidean argmin is point 1, manhattan argmin is point 0.
    const X = [
      [1.4, 0],
      [0.8, 0.8],
      [-2.2, -0.8],
    ];
    const labels = [0, 0, 0];
    const euclidean = await select_medoids(X, labels, 1, 'euclidean');
    expect(euclidean.indices[0]).toBe(1);
    const manhattan = await select_medoids(X, labels, 1, 'manhattan');
    expect(manhattan.indices[0]).toBe(0);
  });

  it('selects the cosine medoid by angle, ignoring magnitude', async () => {
    // Cluster mean is (2, ~1.97); point 2 is euclidean-farthest but almost
    // exactly mean-aligned in direction, so cosine picks it and euclidean
    // does not.
    const X = [
      [1, 0],
      [0, 1],
      [5, 4.9],
    ];
    const labels = [0, 0, 0];
    const euclidean = await select_medoids(X, labels, 1, 'euclidean');
    expect(euclidean.indices[0]).toBe(0);
    const cosine = await select_medoids(X, labels, 1, 'cosine');
    expect(cosine.indices[0]).toBe(2);
  });

  it('breaks exact distance ties towards the lowest sample index', async () => {
    // Both points sit at distance 1 from the mean (0, 0).
    const X = [
      [-1, 0],
      [1, 0],
    ];
    const { indices } = await select_medoids(X, [0, 0], 1, 'euclidean');
    expect(indices[0]).toBe(0);
  });

  it('accepts tensor data and (float) tensor labels', async () => {
    const X = [
      [0, 0],
      [0.1, 0],
      [10, 10],
      [10.1, 10],
    ];
    const labels = [0, 0, 1, 1];
    const from_arrays = await select_medoids(X, labels, 2, 'euclidean');

    const X_tensor = tf.tensor2d(X);
    const labels_tensor = tf.tensor1d(labels, 'float32');
    const from_tensors = await select_medoids(
      X_tensor,
      labels_tensor,
      2,
      'euclidean',
    );
    X_tensor.dispose();
    labels_tensor.dispose();

    expect(Array.from(from_tensors.indices)).toEqual(
      Array.from(from_arrays.indices),
    );
    expect(Array.from(from_tensors.distances)).toEqual(
      Array.from(from_arrays.distances),
    );
  });

  const files = fs
    .readdirSync(FIXTURE_DIR)
    .filter((f) => f.startsWith('medoids_'));

  for (const file of files) {
    const fixture = JSON.parse(
      fs.readFileSync(path.join(FIXTURE_DIR, file), 'utf-8'),
    ) as MedoidFixture;

    it(`matches sklearn-derived medoid indices for ${file}`, async () => {
      const { indices } = await select_medoids(
        fixture.X,
        fixture.labels,
        fixture.params.n_clusters,
        fixture.params.metric,
      );

      for (let c = 0; c < fixture.params.n_clusters; c++) {
        const expected = fixture.medoid_indices[c];
        const got = indices[c];
        if (expected === -1) {
          expect(got).toBe(-1);
          continue;
        }
        // Resolve ties via the stored per-member distances to the mean: the
        // selected medoid must sit at (within tolerance of) the minimum.
        const dists = fixture.distances_to_mean[c];
        const members = fixture.member_indices[c];
        const min_dist = Math.min(...dists);
        const got_local = members.indexOf(got);
        expect(got_local).toBeGreaterThanOrEqual(0);
        expect(dists[got_local]).toBeCloseTo(min_dist, 5);
      }
    });
  }
});

describe('estimator ClusterRepresentations wiring', () => {
  const X = [
    [0, 0],
    [0.1, 0.1],
    [0.2, 0],
    [10, 10],
    [10.1, 9.9],
    [9.9, 10.1],
  ];

  it('AgglomerativeClustering.compute_medoids populates medoid_indices_', async () => {
    const model = new AgglomerativeClustering({ n_clusters: 2, linkage: 'average' });
    await model.fit(X);
    const medoids = await model.compute_medoids(X);
    expect(model.medoid_indices_).toBe(medoids);
    expect(medoids.length).toBe(2);
    for (const idx of medoids) {
      expect(idx).toBeGreaterThanOrEqual(0);
      expect(idx).toBeLessThan(X.length);
    }
  });

  it('AgglomerativeClustering.compute_medoids throws before fit', async () => {
    const model = new AgglomerativeClustering({ n_clusters: 2 });
    await expect(model.compute_medoids(X)).rejects.toThrow();
  });

  it('SpectralClustering exposes medoids via the same surface', async () => {
    const model = new SpectralClustering({ n_clusters: 2, affinity: 'rbf', random_state: 0 });
    await model.fit(X);
    const medoids = await model.compute_medoids(X);
    expect(model.medoid_indices_).toBe(medoids);
    expect(medoids.length).toBe(2);
    for (const idx of medoids) {
      expect(idx).toBeGreaterThanOrEqual(0);
      expect(idx).toBeLessThan(X.length);
    }
    model.dispose();
  });
});
