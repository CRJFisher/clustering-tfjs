import type {
  BaseClustering,
  DataMatrix,
  LabelVector,
  KMeansParams,
} from "./types";
import * as tf from "@tensorflow/tfjs-node";

/**
 * Extremely lightweight – yet reasonably efficient – K-Means implementation
 * intended solely as an internal helper for SpectralClustering.
 *
 * The class purposefully **does not** try to match the full scikit-learn API
 * but merely exposes the minimal surface required by downstream tasks.
 */
export class KMeans implements BaseClustering<KMeansParams> {
  public readonly params: KMeansParams;

  /** Lazily populated labels after calling {@link fit}. */
  public labels_: LabelVector | null = null;

  /** Final cluster centroids (shape: nClusters × nFeatures). */
  public centroids_: tf.Tensor2D | null = null;

  /** Final value of the inertia criterion (sum of squared distances). */
  public inertia_: number | null = null;

  // Reasonable defaults mirroring scikit-learn
  private static readonly DEFAULT_MAX_ITER = 300;
  private static readonly DEFAULT_TOL = 1e-4;
  private static readonly DEFAULT_N_INIT = 1;

  constructor(params: KMeansParams) {
    this.params = { ...params };
    KMeans.validateParams(this.params);
  }

  /* --------------------------------------------------------------------- */
  /*                               Internals                                */
  /* --------------------------------------------------------------------- */

  /** Performs a single deterministic PRNG step – Linear Congruential Gen. */
  private static makeRandomFn(seed?: number): () => number {
    if (seed === undefined) return Math.random;

    let s = Math.floor(seed) % 2147483647;
    if (s <= 0) s += 2147483646;
    return () => {
      s = (s * 16807) % 2147483647;
      return (s - 1) / 2147483646;
    };
  }

  private static validateParams(params: KMeansParams): void {
    const { nClusters, maxIter, tol, nInit } = params;

    if (!Number.isInteger(nClusters) || nClusters < 1) {
      throw new Error("nClusters must be a positive integer (>= 1).");
    }

    if (maxIter !== undefined && (!Number.isInteger(maxIter) || maxIter < 1)) {
      throw new Error("maxIter must be a positive integer (>= 1) when given.");
    }

    if (tol !== undefined && (typeof tol !== "number" || tol < 0)) {
      throw new Error("tol must be a non-negative number when given.");
    }
    if (nInit !== undefined && (!Number.isInteger(nInit) || nInit < 1)) {
      throw new Error("nInit must be a positive integer (>= 1) when given.");
    }
  }

  /* --------------------------------------------------------------------- */
  /*                                   API                                  */
  /* --------------------------------------------------------------------- */

  async fit(X: DataMatrix): Promise<void> {
    // Convert to a Tensor2D of dtype float32 – keep original around for
    // potential multiple initialisations.
    const Xtensor: tf.Tensor2D = (
      X instanceof tf.Tensor ? (X as tf.Tensor2D) : tf.tensor2d(X as number[][], undefined, "float32")
    ).clone();

    const [nSamples, nFeatures] = Xtensor.shape;

    if (nSamples === 0) {
      throw new Error("Input data must contain at least one sample.");
    }

    const K = this.params.nClusters;
    if (K > nSamples) {
      throw new Error("nClusters cannot exceed number of samples.");
    }

    const nInit = this.params.nInit ?? KMeans.DEFAULT_N_INIT;

    // Pre-compute helper structures reused across inits
    const pointsArr: number[][] = (await Xtensor.array()) as number[][];

    // Store best solution across runs
    let bestInertia = Number.POSITIVE_INFINITY;
    let bestLabels: Int32Array | null = null;
    let bestCentroids: tf.Tensor2D | null = null;

    const maxIter = this.params.maxIter ?? KMeans.DEFAULT_MAX_ITER;
    const tol = this.params.tol ?? KMeans.DEFAULT_TOL;

    const baseSeed = this.params.randomState;

    const runOnce = async (seedOffset: number): Promise<{
      inertia: number;
      labels: Int32Array;
      centroids: tf.Tensor2D;
    }> => {
      const rand = KMeans.makeRandomFn(baseSeed !== undefined ? baseSeed + seedOffset : undefined);

      // ----------------------- k-means++ seeding ----------------------- //
      const centroidIdxs: number[] = [];
      centroidIdxs.push(Math.floor(rand() * nSamples));

      while (centroidIdxs.length < K) {
        const distances: number[] = pointsArr.map((p, idx) => {
          if (centroidIdxs.includes(idx)) return 0;
          let minD2 = Number.POSITIVE_INFINITY;
          for (const cIdx of centroidIdxs) {
            const c = pointsArr[cIdx];
            let d2 = 0;
            for (let j = 0; j < nFeatures; j++) {
              const diff = p[j] - c[j];
              d2 += diff * diff;
            }
            if (d2 < minD2) minD2 = d2;
          }
          return minD2;
        });

        const sum = distances.reduce((a, b) => a + b, 0);
        if (sum === 0) {
          for (let i = 0; i < nSamples; i++) {
            if (!centroidIdxs.includes(i)) {
              centroidIdxs.push(i);
              break;
            }
          }
          continue;
        }

        const r = rand() * sum;
        let cumulative = 0;
        let chosen = 0;
        for (let i = 0; i < nSamples; i++) {
          cumulative += distances[i];
          if (r <= cumulative) {
            chosen = i;
            break;
          }
        }
        centroidIdxs.push(chosen);
      }

      let centroids = tf.tensor2d(centroidIdxs.map((i) => pointsArr[i]), [K, nFeatures], "float32");

      let prevInertia = Number.POSITIVE_INFINITY;
      let labels: Int32Array = new Int32Array(nSamples);

      for (let iter = 0; iter < maxIter; iter++) {
        const distances = tf.tidy(() => {
          const xNorm = Xtensor.square().sum(1).reshape([nSamples, 1]);
          const cNorm = centroids.square().sum(1).reshape([1, K]);
          const cross = tf.matMul(Xtensor, centroids.transpose());
          return xNorm.add(cNorm).sub(cross.mul(2));
        });

        labels = (await distances.argMin(1).data()) as Int32Array;

        const minDistSq = await distances.min(1).data();
        const inertia = Array.from(minDistSq as Float32Array).reduce((a, b) => a + b, 0);
        distances.dispose();

        const newCentroidsArr: number[][] = Array.from({ length: K }, () => Array(nFeatures).fill(0));
        const counts: number[] = Array(K).fill(0);

        for (let i = 0; i < nSamples; i++) {
          const label = labels[i];
          counts[label]++;
          const row = pointsArr[i];
          for (let j = 0; j < nFeatures; j++) {
            newCentroidsArr[label][j] += row[j];
          }
        }

        for (let kIdx = 0; kIdx < K; kIdx++) {
          if (counts[kIdx] === 0) {
            newCentroidsArr[kIdx] = Array.from(
              await centroids.slice([kIdx, 0], [1, nFeatures]).array()
            )[0] as number[];
            continue;
          }
          for (let j = 0; j < nFeatures; j++) {
            newCentroidsArr[kIdx][j] /= counts[kIdx];
          }
        }

        const newCentroids = tf.tensor2d(newCentroidsArr, [K, nFeatures], "float32");

        const centroidShift = (await centroids.sub(newCentroids).abs().max().data())[0];

        centroids.dispose();
        centroids = newCentroids;

        const relativeDiff = Math.abs(prevInertia - inertia) / (prevInertia || 1);
        if (relativeDiff <= tol || centroidShift <= tol) {
          prevInertia = inertia;
          break;
        }
        prevInertia = inertia;
      }

      return { inertia: prevInertia, labels, centroids };
    };

    for (let run = 0; run < nInit; run++) {
      const { inertia, labels, centroids } = await runOnce(run);
      if (inertia < bestInertia) {
        if (bestCentroids) bestCentroids.dispose();
        bestInertia = inertia;
        bestLabels = labels;
        bestCentroids = centroids;
      } else {
        // dispose unused centroids to avoid leaks
        centroids.dispose();
      }
    }

    // Save best solution to instance
    this.centroids_ = bestCentroids!;
    this.labels_ = Array.from(bestLabels!);
    this.inertia_ = bestInertia;

    Xtensor.dispose();
  }

  async fitPredict(X: DataMatrix): Promise<LabelVector> {
    await this.fit(X);
    if (this.labels_ == null) {
      throw new Error("KMeans.fit did not compute labels.");
    }
    return this.labels_;
  }
}
