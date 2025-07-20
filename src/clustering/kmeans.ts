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
  // scikit-learn defaults to 10 initialisations which results in more
  // stable solutions, especially for small ambiguous datasets.  Matching
  // the reference implementation improves parity for downstream spectral
  // clustering tests.
  private static readonly DEFAULT_N_INIT = 10;

  constructor(params: KMeansParams) {
    this.params = { ...params };
    KMeans.validateParams(this.params);
  }

  /* --------------------------------------------------------------------- */
  /*                               Internals                                */
  /* --------------------------------------------------------------------- */

  /** Provides deterministic or non-deterministic random stream aligned with NumPy. */
  private static makeRandomStream(seed?: number) {
    const { make_random_stream } = require("../utils/rng");
    return make_random_stream(seed);
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

    // Pre-compute helper structures reused across inits (use full precision
    // original data when available to avoid float32 rounding affecting
    // k-means++ probabilities).

    const pointsArr: number[][] = Array.isArray(X)
      ? (X as number[][])
      : ((await Xtensor.array()) as number[][]);

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
      const randStream = KMeans.makeRandomStream(
        baseSeed !== undefined ? baseSeed + seedOffset : undefined,
      );

      const rand = randStream.rand;

      // ----------------------- k-means++ seeding ----------------------- //
      const centroidIdxs: number[] = [];
      centroidIdxs.push(randStream.randInt(nSamples));

      while (centroidIdxs.length < K) {
        // 1) Compute squared distance to nearest existing centroid for each point
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

        const currentPot = distances.reduce((a, b) => a + b, 0);
        if (currentPot === 0) {
          // All remaining points identical to existing centroids – pick first unused index deterministically
          for (let i = 0; i < nSamples; i++) {
            if (!centroidIdxs.includes(i)) {
              centroidIdxs.push(i);
              break;
            }
          }
          continue;
        }

        // 2) Sample candidate indices according to probability proportional to distance^2
        const localTrials = 2 + Math.floor(Math.log(K));
        const cumulativeDistances: number[] = [];
        let cumSum = 0;
        for (let d of distances) {
          cumSum += d;
          cumulativeDistances.push(cumSum);
        }

        const candidates: number[] = [];
        for (let t = 0; t < localTrials; t++) {
          const r = rand() * currentPot;
          // binary search
          let lo = 0;
          let hi = nSamples - 1;
          while (lo < hi) {
            const mid = Math.floor((lo + hi) / 2);
            if (r <= cumulativeDistances[mid]) {
              hi = mid;
            } else {
              lo = mid + 1;
            }
          }
          candidates.push(lo);
        }

        // 3) Compute potential for each candidate and choose best
        let bestCandidate = candidates[0];
        let bestPotential = Number.POSITIVE_INFINITY;

        for (const cand of candidates) {
          let pot = 0;
          const candPoint = pointsArr[cand];
          for (let i = 0; i < nSamples; i++) {
            const p = pointsArr[i];
            let d2 = 0;
            for (let j = 0; j < nFeatures; j++) {
              const diff = p[j] - candPoint[j];
              d2 += diff * diff;
            }
            const minD2 = Math.min(distances[i], d2);
            pot += minD2;
          }
          if (pot < bestPotential) {
            bestPotential = pot;
            bestCandidate = cand;
          }
        }

        centroidIdxs.push(bestCandidate);
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

        // Handle empty clusters using sklearn's strategy
        const emptyClusters: number[] = [];
        for (let kIdx = 0; kIdx < K; kIdx++) {
          if (counts[kIdx] === 0) {
            emptyClusters.push(kIdx);
            // Keep old centroid temporarily
            newCentroidsArr[kIdx] = Array.from(
              await centroids.slice([kIdx, 0], [1, nFeatures]).array()
            )[0] as number[];
          } else {
            for (let j = 0; j < nFeatures; j++) {
              newCentroidsArr[kIdx][j] /= counts[kIdx];
            }
          }
        }
        
        // If there are empty clusters, reassign them to points farthest from their nearest center
        if (emptyClusters.length > 0) {
          // Compute distances from all points to their nearest center
          const distToNearest = new Float32Array(nSamples);
          for (let i = 0; i < nSamples; i++) {
            distToNearest[i] = minDistSq[i];
          }
          
          // Find indices of points with largest distances
          const indices = Array.from({length: nSamples}, (_, i) => i);
          indices.sort((a, b) => distToNearest[b] - distToNearest[a]);
          
          // Assign farthest points as new centers for empty clusters
          for (let i = 0; i < emptyClusters.length && i < nSamples; i++) {
            const farthestIdx = indices[i];
            const emptyClusterIdx = emptyClusters[i];
            newCentroidsArr[emptyClusterIdx] = [...pointsArr[farthestIdx]];
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
