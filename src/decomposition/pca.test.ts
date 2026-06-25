import fs from 'fs';
import path from 'path';

import { PCA } from '..';
import { power_iteration_eig, unit_init_vector } from './pca';
import * as tf from '../../test_support/tensorflow_helper';

const FIXTURE_DIR = path.join(process.cwd(), '__fixtures__', 'pca');

interface PCAFixture {
  n_components: number;
  X: number[][];
  components_: number[][];
  explained_variance_: number[];
  mean_: number[];
  transform: number[][];
}

/** Per-component sign that aligns `mine` to `ref` (components match up to sign). */
function alignment_signs(mine: number[][], ref: number[][]): number[] {
  return mine.map((row, c) => {
    let dot = 0;
    for (let j = 0; j < row.length; j++) dot += row[j] * ref[c][j];
    return dot < 0 ? -1 : 1;
  });
}

describe('PCA – parity with scikit-learn (svd_flip)', () => {
  const files = fs.readdirSync(FIXTURE_DIR).filter((f) => f.endsWith('.json'));

  for (const file of files) {
    const fixture = JSON.parse(
      fs.readFileSync(path.join(FIXTURE_DIR, file), 'utf-8'),
    ) as PCAFixture;

    it(`matches components, variance, and transform for ${file}`, () => {
      const pca = new PCA({ n_components: fixture.n_components, random_state: 0 });
      const transformed = pca.fit_transform(fixture.X);

      for (let j = 0; j < fixture.mean_.length; j++) {
        expect(pca.mean_![j]).toBeCloseTo(fixture.mean_[j], 5);
      }

      for (let c = 0; c < fixture.n_components; c++) {
        expect(pca.explained_variance_![c]).toBeCloseTo(
          fixture.explained_variance_[c],
          3,
        );
      }

      const signs = alignment_signs(pca.components_!, fixture.components_);
      for (let c = 0; c < fixture.n_components; c++) {
        for (let j = 0; j < pca.components_![c].length; j++) {
          expect(pca.components_![c][j] * signs[c]).toBeCloseTo(
            fixture.components_[c][j],
            3,
          );
        }
      }

      for (let i = 0; i < transformed.length; i++) {
        for (let c = 0; c < fixture.n_components; c++) {
          expect(transformed[i][c] * signs[c]).toBeCloseTo(
            fixture.transform[i][c],
            3,
          );
        }
      }
    });
  }
});

describe('PCA – API behaviour', () => {
  const X = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 10],
    [2, 0, 1],
  ];

  it('rejects n_components greater than the number of features', () => {
    const pca = new PCA({ n_components: 4 });
    expect(() => pca.fit(X)).toThrow();
  });

  it('fit_predict throws (PCA is not a clusterer)', () => {
    const pca = new PCA({ n_components: 2 });
    expect(() => pca.fit_predict(X)).toThrow();
  });

  it('inverse_transform approximately reconstructs full-rank data', () => {
    const pca = new PCA({ n_components: 3, random_state: 0 });
    const z = pca.fit_transform(X);
    const back = pca.inverse_transform(z);
    for (let i = 0; i < X.length; i++) {
      for (let j = 0; j < X[i].length; j++) {
        expect(back[i][j]).toBeCloseTo(X[i][j], 4);
      }
    }
  });

  it('round-trips through to_json/from_json with identical transform', () => {
    const pca = new PCA({ n_components: 2, random_state: 0 });
    pca.fit(X);
    const json = pca.to_json();
    const restored = PCA.from_json(JSON.parse(JSON.stringify(json)));

    expect(restored.components_).toEqual(pca.components_);
    expect(restored.explained_variance_).toEqual(pca.explained_variance_);
    expect(restored.mean_).toEqual(pca.mean_);

    const a = pca.transform(X);
    const b = restored.transform(X);
    expect(b).toEqual(a);
  });

  it('rejects a non-integer or non-positive n_components', () => {
    expect(() => new PCA({ n_components: 0 })).toThrow('positive integer');
    expect(() => new PCA({ n_components: 1.5 })).toThrow('positive integer');
    expect(() => new PCA({ n_components: -2 })).toThrow('positive integer');
  });

  it('rejects empty input', () => {
    expect(() => new PCA({ n_components: 1 }).fit([])).toThrow(
      'at least one sample',
    );
  });

  it('transform, inverse_transform, and to_json throw before fit', () => {
    const pca = new PCA({ n_components: 2 });
    expect(() => pca.transform(X)).toThrow('before fit');
    expect(() => pca.inverse_transform([[0, 0]])).toThrow('before fit');
    expect(() => pca.to_json()).toThrow('before fit');
  });

  it('fits a single sample without NaN (denominator clamps to 1)', () => {
    const pca = new PCA({ n_components: 1, random_state: 0 });
    pca.fit([[1, 2, 3]]);
    expect(pca.mean_).toEqual([1, 2, 3]);
    for (const v of pca.explained_variance_!) {
      expect(Number.isFinite(v)).toBe(true);
      expect(v).toBeCloseTo(0, 10);
    }
  });

  it('handles constant features (exact zeros in centered data)', () => {
    // Column 0 is constant, so every centered value in it is exactly 0.
    const data = [
      [3, 1],
      [3, 5],
      [3, 9],
    ];
    const pca = new PCA({ n_components: 2, random_state: 0 });
    const z = pca.fit_transform(data);
    expect(pca.explained_variance_![0]).toBeCloseTo(16, 6);
    expect(pca.explained_variance_![1]).toBeCloseTo(0, 6);
    expect(z.length).toBe(3);
  });

  it('accepts tensor input and matches the array-input fit', () => {
    const from_array = new PCA({ n_components: 2, random_state: 0 });
    from_array.fit(X);

    const points = tf.tensor2d(X);
    const from_tensor = new PCA({ n_components: 2, random_state: 0 });
    from_tensor.fit(points);
    const transformed = from_tensor.transform(tf.tensor2d([[1, 1, 1]]));
    points.dispose();

    expect(from_tensor.components_).toEqual(from_array.components_);
    expect(transformed).toEqual(from_array.transform([[1, 1, 1]]));
  });
});

describe('power_iteration_eig – degenerate matrices', () => {
  it('falls back to a basis vector when the init candidate has zero norm', () => {
    expect(unit_init_vector([0, 0, 0], 0)).toEqual([1, 0, 0]);
    expect(unit_init_vector([0, 0, 0], 4)).toEqual([0, 1, 0]);
    expect(unit_init_vector([0, 3, 4], 0)).toEqual([0, 0.6, 0.8]);
  });

  it('returns zero eigenvalues for the zero matrix without iterating forever', () => {
    const { components, eigenvalues } = power_iteration_eig(
      [
        [0, 0],
        [0, 0],
      ],
      2,
      0,
    );
    expect(eigenvalues).toEqual([0, 0]);
    expect(components).toHaveLength(2);
    for (const v of components) {
      const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
      expect(norm).toBeCloseTo(1, 12);
    }
  });

  it('clamps the component count to the matrix dimension', () => {
    const { eigenvalues } = power_iteration_eig([[4]], 3, 0);
    expect(eigenvalues).toHaveLength(1);
    expect(eigenvalues[0]).toBeCloseTo(4, 12);
  });
});

describe('power_iteration_eig – known eigenstructure', () => {
  it('recovers eigenvalues (descending) and orthonormal eigenvectors', () => {
    // [[2,1],[1,2]] has eigenvalues 3 (along [1,1]) and 1 (along [1,-1]).
    const { components, eigenvalues } = power_iteration_eig(
      [
        [2, 1],
        [1, 2],
      ],
      2,
      0,
    );
    expect(eigenvalues[0]).toBeCloseTo(3, 6);
    expect(eigenvalues[1]).toBeCloseTo(1, 6);

    // Each eigenvector is unit-norm and the two are mutually orthogonal.
    for (const v of components) {
      expect(Math.hypot(v[0], v[1])).toBeCloseTo(1, 6);
    }
    const dot = components[0][0] * components[1][0] + components[0][1] * components[1][1];
    expect(dot).toBeCloseTo(0, 6);

    // Leading eigenvector points along ±[1,1]/√2.
    expect(Math.abs(components[0][0])).toBeCloseTo(Math.SQRT1_2, 6);
    expect(Math.abs(components[0][1])).toBeCloseTo(Math.SQRT1_2, 6);
  });

  it('is deterministic for a fixed random_state', () => {
    const matrix = [
      [5, 2, 0],
      [2, 5, 0],
      [0, 0, 1],
    ];
    const a = power_iteration_eig(matrix, 3, 7);
    const b = power_iteration_eig(matrix, 3, 7);
    expect(b.eigenvalues).toEqual(a.eigenvalues);
    expect(b.components).toEqual(a.components);
  });
});
