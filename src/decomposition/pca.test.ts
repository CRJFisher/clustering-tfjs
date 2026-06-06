import fs from 'fs';
import path from 'path';

import { PCA } from '..';

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

      // mean
      for (let j = 0; j < fixture.mean_.length; j++) {
        expect(pca.mean_![j]).toBeCloseTo(fixture.mean_[j], 5);
      }

      // explained variance (sign-independent)
      for (let c = 0; c < fixture.n_components; c++) {
        expect(pca.explained_variance_![c]).toBeCloseTo(
          fixture.explained_variance_[c],
          3,
        );
      }

      // components up to per-axis sign (align mine to the sklearn reference)
      const signs = alignment_signs(pca.components_!, fixture.components_);
      for (let c = 0; c < fixture.n_components; c++) {
        for (let j = 0; j < pca.components_![c].length; j++) {
          expect(pca.components_![c][j] * signs[c]).toBeCloseTo(
            fixture.components_[c][j],
            3,
          );
        }
      }

      // transform under the same per-component sign alignment
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
});
