import type { CoreClusteringParams, DataMatrix } from '../clustering/types';
import { is_tensor } from '../tensor/tensor_guards';
import { make_random_stream } from '../random';

/**
 * Principal Component Analysis matching scikit-learn's `PCA(svd_solver='full')`
 * numerically (up to per-component sign).
 *
 * The TensorFlow.js backend has no eigendecomposition, so the leading
 * components are found by power iteration with deflation on the (mean-centered)
 * covariance matrix. The same power-iteration core seeds SOM `'linear'`/`'pca'`
 * weight initialization, keeping map seeding and public reduction numerically
 * identical.
 */

export interface PCAParams extends CoreClusteringParams {
  /** Must be ≤ n_features. */
  n_components: number;
}

export interface PCAJSON {
  params: PCAParams;
  components_: number[][];
  explained_variance_: number[];
  mean_: number[];
}

export interface EigResult {
  /** Eigenvectors as rows, most-significant first. */
  components: number[][];
  /** Corresponding eigenvalues, descending. */
  eigenvalues: number[];
}

function to_array(X: DataMatrix): number[][] {
  return is_tensor(X)
    ? (X.arraySync() as number[][])
    : (X as number[][]);
}

/**
 * Unit-norm power-iteration start vector for component `c`: the candidate
 * normalized, or the `c % d` standard basis vector when the candidate has
 * zero norm (so iteration always starts from a valid direction).
 */
function unit_init_vector(candidate: number[], c: number): number[] {
  const d = candidate.length;
  const vn = Math.sqrt(candidate.reduce((s, x) => s + x * x, 0));
  if (vn === 0) {
    return candidate.map((_x, i) => (i === c % d ? 1 : 0));
  }
  return candidate.map((x) => x / vn);
}

/**
 * Top-`k` eigenvectors/eigenvalues of a symmetric matrix via power iteration
 * with deflation. Deterministic for a fixed `random_state`. Shared by
 * {@link PCA} and SOM principal-component initialization.
 *
 */
export function power_iteration_eig(
  matrix: number[][],
  k: number,
  random_state?: number,
): EigResult {
  const d = matrix.length;
  const count = Math.min(k, d);
  const rng = make_random_stream(random_state);

  const m: number[][] = matrix.map((row) => [...row]);
  const components: number[][] = [];
  const eigenvalues: number[] = [];

  const mat_vec = (mtx: number[][], v: number[]): number[] => {
    const out = new Array<number>(d).fill(0);
    for (let i = 0; i < d; i++) {
      const row = mtx[i];
      let s = 0;
      for (let j = 0; j < d; j++) s += row[j] * v[j];
      out[i] = s;
    }
    return out;
  };
  const norm = (v: number[]): number => Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  const dot = (a: number[], b: number[]): number =>
    a.reduce((s, x, i) => s + x * b[i], 0);

  for (let c = 0; c < count; c++) {
    let v = unit_init_vector(
      Array.from({ length: d }, () => rng.rand() * 2 - 1),
      c,
    );

    for (let iter = 0; iter < 300; iter++) {
      const w = mat_vec(m, v);
      const nw = norm(w);
      if (nw < 1e-12) break;
      const next = w.map((x) => x / nw);
      if (Math.abs(Math.abs(dot(next, v)) - 1) < 1e-12) {
        v = next;
        break;
      }
      v = next;
    }

    const eigenvalue = dot(v, mat_vec(m, v));
    components.push(v);
    eigenvalues.push(eigenvalue);

    // Deflate: m -= eigenvalue * v vᵀ
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) {
        m[i][j] -= eigenvalue * v[i] * v[j];
      }
    }
  }

  return { components, eigenvalues };
}

export class PCA {
  public readonly params: PCAParams;

  /** Principal axes as rows (`n_components × n_features`). Null until `fit`. */
  public components_: number[][] | null = null;
  /** Variance explained by each component. Null until `fit`. */
  public explained_variance_: number[] | null = null;
  /** Per-feature empirical mean used for centering. Null until `fit`. */
  public mean_: number[] | null = null;

  constructor(params: PCAParams) {
    this.params = { ...params };
    if (!Number.isInteger(params.n_components) || params.n_components < 1) {
      throw new Error('n_components must be a positive integer (>= 1).');
    }
  }

  fit(X: DataMatrix): this {
    const data = to_array(X);
    const n = data.length;
    if (n === 0) throw new Error('Input data must contain at least one sample.');
    const d = data[0].length;
    if (this.params.n_components > d) {
      throw new Error(
        `n_components (${this.params.n_components}) cannot exceed the number of features (${d}).`,
      );
    }

    const mean = new Array<number>(d).fill(0);
    for (const row of data) for (let j = 0; j < d; j++) mean[j] += row[j];
    for (let j = 0; j < d; j++) mean[j] /= n;
    const centered = data.map((row) => row.map((v, j) => v - mean[j]));

    // Covariance = Xcᵀ Xc / (n - 1) (sklearn convention).
    const denom = n > 1 ? n - 1 : 1;
    const cov: number[][] = Array.from({ length: d }, () =>
      new Array<number>(d).fill(0),
    );
    for (const row of centered) {
      for (let i = 0; i < d; i++) {
        const ri = row[i];
        if (ri === 0) continue;
        for (let j = i; j < d; j++) {
          cov[i][j] += ri * row[j];
        }
      }
    }
    for (let i = 0; i < d; i++) {
      for (let j = i; j < d; j++) {
        cov[i][j] /= denom;
        cov[j][i] = cov[i][j];
      }
    }

    const { components, eigenvalues } = power_iteration_eig(
      cov,
      this.params.n_components,
      this.params.random_state,
    );

    this.components_ = components;
    this.explained_variance_ = eigenvalues;
    this.mean_ = mean;
    return this;
  }

  transform(X: DataMatrix): number[][] {
    if (this.components_ == null || this.mean_ == null) {
      throw new Error('PCA.transform called before fit().');
    }
    const data = to_array(X);
    const mean = this.mean_;
    const comps = this.components_;
    return data.map((row) => {
      const centered = row.map((v, j) => v - mean[j]);
      return comps.map((comp) => {
        let s = 0;
        for (let j = 0; j < comp.length; j++) s += centered[j] * comp[j];
        return s;
      });
    });
  }

  fit_transform(X: DataMatrix): number[][] {
    this.fit(X);
    return this.transform(X);
  }

  inverse_transform(Z: DataMatrix): number[][] {
    if (this.components_ == null || this.mean_ == null) {
      throw new Error('PCA.inverse_transform called before fit().');
    }
    const data = to_array(Z);
    const mean = this.mean_;
    const comps = this.components_;
    const d = mean.length;
    return data.map((z) => {
      const out = new Array<number>(d).fill(0);
      for (let c = 0; c < comps.length; c++) {
        const zc = z[c];
        const comp = comps[c];
        for (let j = 0; j < d; j++) out[j] += zc * comp[j];
      }
      for (let j = 0; j < d; j++) out[j] += mean[j];
      return out;
    });
  }

  /** PCA is a dimensionality reducer, not a clusterer. */
  fit_predict(_X: DataMatrix): never {
    throw new Error('PCA does not support fit_predict; it is not a clusterer.');
  }

  to_json(): PCAJSON {
    if (
      this.components_ == null ||
      this.explained_variance_ == null ||
      this.mean_ == null
    ) {
      throw new Error('PCA.to_json called before fit().');
    }
    return {
      params: { ...this.params },
      components_: this.components_.map((row) => [...row]),
      explained_variance_: [...this.explained_variance_],
      mean_: [...this.mean_],
    };
  }

  static from_json(json: PCAJSON): PCA {
    const pca = new PCA(json.params);
    pca.components_ = json.components_.map((row) => [...row]);
    pca.explained_variance_ = [...json.explained_variance_];
    pca.mean_ = [...json.mean_];
    return pca;
  }
}
