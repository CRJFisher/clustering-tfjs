import type {
  BaseClustering,
  DataMatrix,
  KMeansParams,
} from './types';
import * as tf from '../backend/adapter';
import { is_tensor } from '../tensor/tensor_guards';
import { make_random_stream } from '../random';

/**
 * K-Means clustering algorithm using Lloyd's iteration with K-means++ initialization.
 *
 * Supports multiple random initializations (`n_init`) and selects the solution
 * with the lowest inertia, matching scikit-learn's default behavior.
 */
export class KMeans implements BaseClustering<KMeansParams> {
  public readonly params: KMeansParams;

  /** Lazily populated labels after calling {@link fit}. */
  public labels_: number[] | null = null;

  /** Final cluster centroids (shape: n_clusters × n_features). */
  public centroids_: tf.Tensor2D | null = null;

  /** Final value of the inertia criterion (sum of squared distances). */
  public inertia_: number | null = null;

  // Reasonable defaults mirroring scikit-learn
  private static readonly DEFAULT_MAX_ITER = 300;
  private static readonly DEFAULT_TOL = 1e-4;
  // scikit-learn defaults to 10 initialisations which results in more
  // stable solutions, especially for small ambiguous datasets.  Matching
  // the reference implementation improves parity for downstream spectral
  // clustering tests.
  private static readonly DEFAULT_N_INIT = 10;

  /**
   * @param params - Configuration for K-Means clustering.
   */
  constructor(params: KMeansParams) {
    this.params = { ...params };
    KMeans.validate_params(this.params);
  }

  /* --------------------------------------------------------------------- */
  /*                               Internals                                */
  /* --------------------------------------------------------------------- */

  /** Provides deterministic or non-deterministic random stream aligned with NumPy. */
  private static make_random_stream(seed?: number) {
    return make_random_stream(seed);
  }

  private static validate_params(params: KMeansParams): void {
    const { n_clusters, max_iter, tol, n_init } = params;

    if (!Number.isInteger(n_clusters) || n_clusters < 1) {
      throw new Error('n_clusters must be a positive integer (>= 1).');
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

  /* --------------------------------------------------------------------- */
  /*                                   API                                  */
  /* --------------------------------------------------------------------- */

  /**
   * Disposes any tensors kept as instance state and resets internal caches.
   */
  public dispose(): void {
    if (this.centroids_ != null) {
      this.centroids_.dispose();
      this.centroids_ = null;
    }
    this.labels_ = null;
    this.inertia_ = null;
  }

  /**
   * Fits the K-Means model to the input data.
   *
   * @param X - Input data matrix of shape [n_samples, n_features].
   * @returns A promise that resolves when fitting is complete.
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
    // Validate input dimensions before creating tensors to avoid leaks on throw.
    const n_samples = is_tensor(X) ? (X as tf.Tensor2D).shape[0] : (X as number[][]).length;
    if (n_samples === 0) {
      throw new Error('Input data must contain at least one sample.');
    }
    const K = this.params.n_clusters;
    if (K > n_samples) {
      throw new Error('n_clusters cannot exceed number of samples.');
    }

    // Dispose previous state if the estimator is re-used.
    this.dispose();

    // Convert to a Tensor2D of dtype float32 – keep original around for
    // potential multiple initialisations.
    // When X is already a tensor we clone to avoid mutating the caller's data.
    // When X is a plain array, tf.tensor2d already creates a new tensor.
    const x_tensor: tf.Tensor2D = is_tensor(X)
      ? (X as tf.Tensor2D).clone()
      : tf.tensor2d(X as number[][], undefined, 'float32');

    const [, n_features] = x_tensor.shape;

    const n_init = this.params.n_init ?? KMeans.DEFAULT_N_INIT;

    // Pre-compute helper structures reused across inits (use full precision
    // original data when available to avoid float32 rounding affecting
    // k-means++ probabilities).

    const points_arr: number[][] = Array.isArray(X)
      ? (X as number[][])
      : ((await x_tensor.array()) as number[][]);

    // Store best solution across runs
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

      // ----------------------- k-means++ seeding ----------------------- //
      const centroid_idxs: number[] = [];
      const centroid_set = new Set<number>();
      const first_idx = rand_stream.rand_int(n_samples);
      centroid_idxs.push(first_idx);
      centroid_set.add(first_idx);

      while (centroid_idxs.length < K) {
        // 1) Compute squared distance to nearest existing centroid for each point
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
          // All remaining points identical to existing centroids – pick first unused index deterministically
          for (let i = 0; i < n_samples; i++) {
            if (!centroid_set.has(i)) {
              centroid_idxs.push(i);
              centroid_set.add(i);
              break;
            }
          }
          continue;
        }

        // 2) Sample candidate indices according to probability proportional to distance^2
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
          // binary search
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

        // 3) Compute potential for each candidate and choose best
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

        // Handle empty clusters using sklearn's strategy
        const empty_clusters: number[] = [];
        for (let k_idx = 0; k_idx < K; k_idx++) {
          if (counts[k_idx] === 0) {
            empty_clusters.push(k_idx);
            // Keep old centroid temporarily
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

        // If there are empty clusters, reassign them to points farthest from their nearest center
        if (empty_clusters.length > 0) {
          // Compute distances from all points to their nearest center
          const dist_to_nearest = new Float32Array(n_samples);
          for (let i = 0; i < n_samples; i++) {
            dist_to_nearest[i] = min_dist_sq[i];
          }

          // Find indices of points with largest distances
          const indices = Array.from({ length: n_samples }, (_, i) => i);
          indices.sort((a, b) => dist_to_nearest[b] - dist_to_nearest[a]);

          // Assign farthest points as new centers for empty clusters
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

    // Save best solution to instance
    this.centroids_ = best_centroids!;
    this.labels_ = Array.from(best_labels!);
    this.inertia_ = best_inertia;

    x_tensor.dispose();
  }

  /**
   * Fits the model and returns cluster labels.
   *
   * @param X - Input data matrix of shape [n_samples, n_features].
   * @returns Array of cluster labels for each sample.
   * @throws {Error} If input data is empty or n_clusters exceeds n_samples.
   */
  async fit_predict(X: DataMatrix): Promise<number[]> {
    await this.fit(X);
    if (this.labels_ == null) {
      throw new Error('KMeans.fit did not compute labels.');
    }
    return this.labels_;
  }
}
