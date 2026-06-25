import type {
  BaseClustering,
  DataMatrix,
  KMeansParams,
} from './types';
import * as tf from '../backend/adapter';
import { is_tensor } from '../tensor/tensor_guards';
import { make_random_stream } from '../random';
import { pairwise_distance_matrix } from '../distance/pairwise_distance';

export interface KMeansJSON {
  params: KMeansParams;
  centroids_: number[][];
  inertia_: number | null;
}

// zero-norm rows returned unchanged
function l2_normalize_rows(rows: number[][]): number[][] {
  return rows.map((row) => {
    let norm = 0;
    for (const v of row) norm += v * v;
    norm = Math.sqrt(norm);
    if (norm === 0) return row.slice();
    return row.map((v) => v / norm);
  });
}

/**
 * K-Means clustering algorithm using Lloyd's iteration with K-means++ initialization.
 *
 * Supports multiple random initializations (`n_init`) and selects the solution
 * with the lowest inertia, matching scikit-learn's default behavior.
 */
export class KMeans implements BaseClustering<KMeansParams> {
  public readonly params: KMeansParams;

  public labels_: number[] | null = null;

  public centroids_: tf.Tensor2D | null = null;

  public inertia_: number | null = null;

  private static readonly DEFAULT_MAX_ITER = 300;
  private static readonly DEFAULT_TOL = 1e-4;
  // scikit-learn defaults to 10 initialisations which results in more
  // stable solutions, especially for small ambiguous datasets.  Matching
  // the reference implementation improves parity for downstream spectral
  // clustering tests.
  private static readonly DEFAULT_N_INIT = 10;

  constructor(params: KMeansParams) {
    this.params = { ...params };
    KMeans.validate_params(this.params);
  }

  private static make_random_stream(seed?: number) {
    return make_random_stream(seed);
  }

  private static validate_params(params: KMeansParams): void {
    const { n_clusters, max_iter, tol, n_init, metric } = params;

    if (!Number.isInteger(n_clusters) || n_clusters < 1) {
      throw new Error('n_clusters must be a positive integer (>= 1).');
    }

    if (metric !== undefined && metric !== 'euclidean' && metric !== 'cosine') {
      throw new Error("metric must be 'euclidean' or 'cosine' when given.");
    }

    if (max_iter !== undefined && (!Number.isInteger(max_iter) || max_iter < 1)) {
      throw new Error('max_iter must be a positive integer (>= 1) when given.');
    }

    if (tol !== undefined && (typeof tol !== 'number' || tol < 0)) {
      throw new Error('tol must be a non-negative number when given.');
    }
    if (n_init !== undefined && (!Number.isInteger(n_init) || n_init < 1)) {
      throw new Error('n_init must be a positive integer (>= 1) when given.');
    }
  }

  public dispose(): void {
    if (this.centroids_ != null) {
      this.centroids_.dispose();
      this.centroids_ = null;
    }
    this.labels_ = null;
    this.inertia_ = null;
  }

  /**
   * @throws {Error} If input data is empty or n_clusters exceeds n_samples.
   *
   * @example
   * ```typescript
   * const kmeans = new KMeans({ n_clusters: 3 });
   * await kmeans.fit([[1, 2], [3, 4], [5, 6]]);
   * console.log(kmeans.labels_);
   * ```
   */
  async fit(X: DataMatrix): Promise<void> {
    if (this.params.metric === 'cosine') {
      await this.fit_cosine(X);
      return;
    }

    // Validate input dimensions before creating tensors to avoid leaks on throw.
    const n_samples = is_tensor(X) ? (X as tf.Tensor2D).shape[0] : (X as number[][]).length;
    if (n_samples === 0) {
      throw new Error('Input data must contain at least one sample.');
    }
    const K = this.params.n_clusters;
    if (K > n_samples) {
      throw new Error('n_clusters cannot exceed number of samples.');
    }

    this.dispose();

    // When X is already a tensor we clone to avoid mutating the caller's data.
    // When X is a plain array, tf.tensor2d already creates a new tensor.
    const x_tensor: tf.Tensor2D = is_tensor(X)
      ? (X as tf.Tensor2D).clone()
      : tf.tensor2d(X as number[][], undefined, 'float32');

    const [, n_features] = x_tensor.shape;

    const n_init = this.params.n_init ?? KMeans.DEFAULT_N_INIT;

    // Use full-precision original data to avoid float32 rounding in k-means++ probabilities.
    const points_arr: number[][] = Array.isArray(X)
      ? (X as number[][])
      : ((await x_tensor.array()) as number[][]);

    let best_inertia = Number.POSITIVE_INFINITY;
    let best_labels: Int32Array | null = null;
    let best_centroids: tf.Tensor2D | null = null;

    const max_iter = this.params.max_iter ?? KMeans.DEFAULT_MAX_ITER;
    const tol = this.params.tol ?? KMeans.DEFAULT_TOL;

    const base_seed = this.params.random_state;

    const run_once = async (
      seed_offset: number,
    ): Promise<{
      inertia: number;
      labels: Int32Array;
      centroids: tf.Tensor2D;
    }> => {
      const rand_stream = KMeans.make_random_stream(
        base_seed !== undefined ? base_seed + seed_offset : undefined,
      );

      const rand = rand_stream.rand;

      const centroid_idxs: number[] = [];
      const centroid_set = new Set<number>();
      const first_idx = rand_stream.rand_int(n_samples);
      centroid_idxs.push(first_idx);
      centroid_set.add(first_idx);

      while (centroid_idxs.length < K) {
        const distances: number[] = points_arr.map((p, idx) => {
          if (centroid_set.has(idx)) return 0;
          let min_d2 = Number.POSITIVE_INFINITY;
          for (const c_idx of centroid_idxs) {
            const c = points_arr[c_idx];
            let d2 = 0;
            for (let j = 0; j < n_features; j++) {
              const diff = p[j] - c[j];
              d2 += diff * diff;
            }
            if (d2 < min_d2) min_d2 = d2;
          }
          return min_d2;
        });

        const current_pot = distances.reduce((a, b) => a + b, 0);
        if (current_pot === 0) {
          for (let i = 0; i < n_samples; i++) {
            if (!centroid_set.has(i)) {
              centroid_idxs.push(i);
              centroid_set.add(i);
              break;
            }
          }
          continue;
        }

        const local_trials = 2 + Math.floor(Math.log(K));
        const cumulative_distances: number[] = [];
        let cum_sum = 0;
        for (const d of distances) {
          cum_sum += d;
          cumulative_distances.push(cum_sum);
        }

        const candidates: number[] = [];
        for (let t = 0; t < local_trials; t++) {
          const r = rand() * current_pot;
          let lo = 0;
          let hi = n_samples - 1;
          while (lo < hi) {
            const mid = Math.floor((lo + hi) / 2);
            if (r <= cumulative_distances[mid]) {
              hi = mid;
            } else {
              lo = mid + 1;
            }
          }
          candidates.push(lo);
        }

        let best_candidate = candidates[0];
        let best_potential = Number.POSITIVE_INFINITY;

        for (const cand of candidates) {
          let pot = 0;
          const cand_point = points_arr[cand];
          for (let i = 0; i < n_samples; i++) {
            const p = points_arr[i];
            let d2 = 0;
            for (let j = 0; j < n_features; j++) {
              const diff = p[j] - cand_point[j];
              d2 += diff * diff;
            }
            const min_d2 = Math.min(distances[i], d2);
            pot += min_d2;
          }
          if (pot < best_potential) {
            best_potential = pot;
            best_candidate = cand;
          }
        }

        centroid_idxs.push(best_candidate);
        centroid_set.add(best_candidate);
      }

      let centroids = tf.tensor2d(
        centroid_idxs.map((i) => points_arr[i]),
        [K, n_features],
        'float32',
      );

      let prev_inertia = Number.POSITIVE_INFINITY;
      let labels: Int32Array = new Int32Array(n_samples);

      for (let iter = 0; iter < max_iter; iter++) {
        const distances = tf.tidy(() => {
          const x_norm = x_tensor.square().sum(1).reshape([n_samples, 1]);
          const c_norm = centroids.square().sum(1).reshape([1, K]);
          const cross = tf.mat_mul(x_tensor, centroids.transpose());
          const dist_sq = x_norm.add(c_norm).sub(cross.mul(2));
          return tf.maximum(dist_sq, tf.scalar(0, 'float32'));
        });

        const arg_min_tensor = distances.argMin(1);
        labels = (await arg_min_tensor.data()) as Int32Array;
        arg_min_tensor.dispose();

        const min_tensor = distances.min(1);
        const min_dist_sq = await min_tensor.data();
        min_tensor.dispose();
        const inertia = Array.from(min_dist_sq as Float32Array).reduce(
          (a, b) => a + b,
          0,
        );
        distances.dispose();

        const new_centroids_arr: number[][] = Array.from({ length: K }, () =>
          Array(n_features).fill(0),
        );
        const counts: number[] = Array(K).fill(0);

        for (let i = 0; i < n_samples; i++) {
          const label = labels[i];
          counts[label]++;
          const row = points_arr[i];
          for (let j = 0; j < n_features; j++) {
            new_centroids_arr[label][j] += row[j];
          }
        }

        const empty_clusters: number[] = [];
        for (let k_idx = 0; k_idx < K; k_idx++) {
          if (counts[k_idx] === 0) {
            empty_clusters.push(k_idx);
            const slice_tensor = centroids.slice([k_idx, 0], [1, n_features]);
            new_centroids_arr[k_idx] = Array.from(
              await slice_tensor.array(),
            )[0] as number[];
            slice_tensor.dispose();
          } else {
            for (let j = 0; j < n_features; j++) {
              new_centroids_arr[k_idx][j] /= counts[k_idx];
            }
          }
        }

        if (empty_clusters.length > 0) {
          const dist_to_nearest = new Float32Array(n_samples);
          for (let i = 0; i < n_samples; i++) {
            dist_to_nearest[i] = min_dist_sq[i];
          }

          const indices = Array.from({ length: n_samples }, (_, i) => i);
          indices.sort((a, b) => dist_to_nearest[b] - dist_to_nearest[a]);

          for (let i = 0; i < empty_clusters.length && i < n_samples; i++) {
            const farthest_idx = indices[i];
            const empty_cluster_idx = empty_clusters[i];
            new_centroids_arr[empty_cluster_idx] = [...points_arr[farthest_idx]];
          }
        }

        const new_centroids = tf.tensor2d(
          new_centroids_arr,
          [K, n_features],
          'float32',
        );

        const shift_tensor = tf.tidy(() => centroids.sub(new_centroids).abs().max());
        const centroid_shift = (await shift_tensor.data())[0];
        shift_tensor.dispose();

        centroids.dispose();
        centroids = new_centroids;

        const relative_diff =
          Math.abs(prev_inertia - inertia) / (prev_inertia || 1);
        if (relative_diff <= tol || centroid_shift <= tol) {
          prev_inertia = inertia;
          break;
        }
        prev_inertia = inertia;
      }

      return { inertia: prev_inertia, labels, centroids };
    };

    for (let run = 0; run < n_init; run++) {
      const { inertia, labels, centroids } = await run_once(run);
      if (inertia < best_inertia) {
        if (best_centroids) best_centroids.dispose();
        best_inertia = inertia;
        best_labels = labels;
        best_centroids = centroids;
      } else {
        // dispose unused centroids to avoid leaks
        centroids.dispose();
      }
    }

    this.centroids_ = best_centroids!;
    this.labels_ = Array.from(best_labels!);
    this.inertia_ = best_inertia;

    x_tensor.dispose();
  }

  /**
   * @throws {Error} If input data is empty or n_clusters exceeds n_samples.
   */
  async fit_predict(X: DataMatrix): Promise<number[]> {
    await this.fit(X);
    if (this.labels_ == null) {
      throw new Error('KMeans.fit did not compute labels.');
    }
    return this.labels_;
  }

  /**
   * Distances are computed with `pairwise_distance_matrix` under the model's
   * metric (cosine rows are L2-normalized first, matching `fit`).
   *
   * @throws {Error} If called before `fit()` has populated `centroids_`.
   */
  async predict(X: DataMatrix): Promise<number[]> {
    if (this.centroids_ == null) {
      throw new Error('KMeans.predict called before fit(); centroids_ is null.');
    }

    const metric = this.params.metric ?? 'euclidean';
    const points_raw: number[][] = is_tensor(X)
      ? ((await (X as tf.Tensor2D).array()) as number[][])
      : (X as number[][]);

    if (points_raw.length === 0) {
      return [];
    }

    const points =
      metric === 'cosine' ? l2_normalize_rows(points_raw) : points_raw;
    const n = points.length;
    const n_features = points[0].length;

    return tf.tidy(() => {
      const centroid_rows = this.centroids_!.arraySync() as number[][];
      const k = centroid_rows.length;
      const combined = tf.tensor2d(
        [...points, ...centroid_rows],
        [n + k, n_features],
        'float32',
      );
      const full = pairwise_distance_matrix(combined, metric).arraySync() as number[][];

      const labels: number[] = new Array<number>(n);
      for (let i = 0; i < n; i++) {
        let best = 0;
        let best_d = full[i][n];
        for (let j = 1; j < k; j++) {
          const d = full[i][n + j];
          if (d < best_d) {
            best_d = d;
            best = j;
          }
        }
        labels[i] = best;
      }
      return labels;
    });
  }

  /**
   * @throws {Error} If the model is unfitted.
   */
  get_centroids(): number[][] {
    if (this.centroids_ == null) {
      throw new Error('KMeans.get_centroids called before fit(); centroids_ is null.');
    }
    return this.centroids_.arraySync() as number[][];
  }

  /**
   * The centroid matrix fully determines cluster assignment; together with the
   * constructor params and `inertia_` it forms a complete snapshot.
   *
   * @throws {Error} If the model is unfitted.
   */
  to_json(): KMeansJSON {
    if (this.centroids_ == null) {
      throw new Error('KMeans.to_json called before fit(); centroids_ is null.');
    }
    return {
      params: { ...this.params },
      centroids_: this.centroids_.arraySync() as number[][],
      inertia_: this.inertia_,
    };
  }

  /**
   * The restored model reproduces cluster assignment via {@link predict} without re-fitting.
   */
  static from_json(json: KMeansJSON): KMeans {
    const model = new KMeans(json.params);
    model.centroids_ = tf.tensor2d(json.centroids_);
    model.inertia_ = json.inertia_;
    return model;
  }

  /**
   * Spherical k-means: L2-normalizes the data onto the unit sphere and runs
   * k-means++ seeding and Lloyd assignment under the cosine metric, routing
   * every distance through `pairwise_distance_matrix(points, 'cosine')`.
   * Centroids are stored as the (un-renormalized) means of the assigned unit
   * vectors, matching the `normalize(X)` + KMeans reference convention.
   */
  private async fit_cosine(X: DataMatrix): Promise<void> {
    const points_raw: number[][] = is_tensor(X)
      ? ((await (X as tf.Tensor2D).array()) as number[][])
      : (X as number[][]);

    const n_samples = points_raw.length;
    if (n_samples === 0) {
      throw new Error('Input data must contain at least one sample.');
    }
    const K = this.params.n_clusters;
    if (K > n_samples) {
      throw new Error('n_clusters cannot exceed number of samples.');
    }

    this.dispose();

    const n_features = points_raw[0].length;
    const points = l2_normalize_rows(points_raw);

    const max_iter = this.params.max_iter ?? KMeans.DEFAULT_MAX_ITER;
    const tol = this.params.tol ?? KMeans.DEFAULT_TOL;
    const n_init = this.params.n_init ?? KMeans.DEFAULT_N_INIT;
    const base_seed = this.params.random_state;

    const d_cos: number[][] = tf.tidy(() => {
      const x = tf.tensor2d(points, [n_samples, n_features], 'float32');
      return pairwise_distance_matrix(x, 'cosine').arraySync() as number[][];
    });

    const cross_cosine = (centroids: number[][]): number[][] => {
      return tf.tidy(() => {
        const combined = tf.tensor2d(
          [...points, ...centroids],
          [n_samples + centroids.length, n_features],
          'float32',
        );
        const full = pairwise_distance_matrix(combined, 'cosine').arraySync() as number[][];
        const out: number[][] = new Array<number[]>(n_samples);
        for (let i = 0; i < n_samples; i++) {
          out[i] = full[i].slice(n_samples, n_samples + centroids.length);
        }
        return out;
      });
    };

    const run_once = (
      seed_offset: number,
    ): { inertia: number; labels: Int32Array; centroids: number[][] } => {
      const rand_stream = KMeans.make_random_stream(
        base_seed !== undefined ? base_seed + seed_offset : undefined,
      );
      const rand = rand_stream.rand;

      const centroid_idxs: number[] = [];
      const centroid_set = new Set<number>();
      const first_idx = rand_stream.rand_int(n_samples);
      centroid_idxs.push(first_idx);
      centroid_set.add(first_idx);

      while (centroid_idxs.length < K) {
        const distances: number[] = points.map((_p, idx) => {
          if (centroid_set.has(idx)) return 0;
          let min_d = Number.POSITIVE_INFINITY;
          for (const c_idx of centroid_idxs) {
            const d = d_cos[idx][c_idx];
            if (d < min_d) min_d = d;
          }
          return min_d * min_d;
        });

        const current_pot = distances.reduce((a, b) => a + b, 0);
        if (current_pot === 0) {
          for (let i = 0; i < n_samples; i++) {
            if (!centroid_set.has(i)) {
              centroid_idxs.push(i);
              centroid_set.add(i);
              break;
            }
          }
          continue;
        }

        const local_trials = 2 + Math.floor(Math.log(K));
        const cumulative: number[] = [];
        let cum = 0;
        for (const d of distances) {
          cum += d;
          cumulative.push(cum);
        }

        const candidates: number[] = [];
        for (let t = 0; t < local_trials; t++) {
          const r = rand() * current_pot;
          let lo = 0;
          let hi = n_samples - 1;
          while (lo < hi) {
            const mid = Math.floor((lo + hi) / 2);
            if (r <= cumulative[mid]) hi = mid;
            else lo = mid + 1;
          }
          candidates.push(lo);
        }

        let best_candidate = candidates[0];
        let best_potential = Number.POSITIVE_INFINITY;
        for (const cand of candidates) {
          let pot = 0;
          for (let i = 0; i < n_samples; i++) {
            const d = d_cos[i][cand];
            pot += Math.min(distances[i], d * d);
          }
          if (pot < best_potential) {
            best_potential = pot;
            best_candidate = cand;
          }
        }

        centroid_idxs.push(best_candidate);
        centroid_set.add(best_candidate);
      }

      let centroids: number[][] = centroid_idxs.map((i) => points[i].slice());
      let prev_inertia = Number.POSITIVE_INFINITY;
      const labels = new Int32Array(n_samples);

      for (let iter = 0; iter < max_iter; iter++) {
        const dist_pc = cross_cosine(centroids);
        let inertia = 0;
        const new_centroids: number[][] = Array.from({ length: K }, () =>
          new Array<number>(n_features).fill(0),
        );
        const counts: number[] = new Array<number>(K).fill(0);
        const min_dist: number[] = new Array<number>(n_samples).fill(0);

        for (let i = 0; i < n_samples; i++) {
          let best = 0;
          let best_d = dist_pc[i][0];
          for (let j = 1; j < K; j++) {
            if (dist_pc[i][j] < best_d) {
              best_d = dist_pc[i][j];
              best = j;
            }
          }
          labels[i] = best;
          min_dist[i] = best_d;
          inertia += best_d * best_d;
          counts[best]++;
          const row = points[i];
          for (let f = 0; f < n_features; f++) new_centroids[best][f] += row[f];
        }

        const empty: number[] = [];
        for (let j = 0; j < K; j++) {
          if (counts[j] === 0) {
            empty.push(j);
            new_centroids[j] = centroids[j].slice();
          } else {
            for (let f = 0; f < n_features; f++) new_centroids[j][f] /= counts[j];
          }
        }
        if (empty.length > 0) {
          const order = Array.from({ length: n_samples }, (_v, i) => i);
          order.sort((a, b) => min_dist[b] - min_dist[a]);
          for (let e = 0; e < empty.length && e < n_samples; e++) {
            new_centroids[empty[e]] = points[order[e]].slice();
          }
        }

        let shift = 0;
        for (let j = 0; j < K; j++) {
          for (let f = 0; f < n_features; f++) {
            const diff = Math.abs(centroids[j][f] - new_centroids[j][f]);
            if (diff > shift) shift = diff;
          }
        }
        centroids = new_centroids;

        const rel = Math.abs(prev_inertia - inertia) / (prev_inertia || 1);
        if (rel <= tol || shift <= tol) {
          prev_inertia = inertia;
          break;
        }
        prev_inertia = inertia;
      }

      return { inertia: prev_inertia, labels, centroids };
    };

    let best_inertia = Number.POSITIVE_INFINITY;
    let best_labels: Int32Array | null = null;
    let best_centroids: number[][] | null = null;

    for (let run = 0; run < n_init; run++) {
      const { inertia, labels, centroids } = run_once(run);
      if (inertia < best_inertia) {
        best_inertia = inertia;
        best_labels = labels;
        best_centroids = centroids;
      }
    }

    this.centroids_ = tf.tensor2d(best_centroids!, [K, n_features], 'float32');
    this.labels_ = Array.from(best_labels!);
    this.inertia_ = best_inertia;
  }
}
