import type {
  DataMatrix,
  LabelVector,
  SpectralClusteringParams,
  BaseClustering,
} from "./types";
import * as tf from "@tensorflow/tfjs-node";

/**
 * Spectral clustering estimator skeleton.
 *
 * This initial implementation only covers:
 *   • Constructor & hyper-parameter validation
 *   • Public instance properties
 *   • Synchronous method stubs for `fit` / `fitPredict`
 *
 * The heavy lifting – affinity matrix construction, graph Laplacian
 * computation, eigen-decomposition and the final k-means step – will be
 * implemented in subsequent tasks (see backlog).
 */
export class SpectralClustering
  implements BaseClustering<SpectralClusteringParams>
{
  /** Hyper-parameters (deep-copied from user input). */
  public readonly params: SpectralClusteringParams;

  /** Lazy-filled cluster labels after calling `fit`. */
  public labels_: LabelVector | null = null;

  /** Cached affinity matrix (shape: nSamples × nSamples). */
  public affinityMatrix_: tf.Tensor2D | null = null;

  // Allowed affinity options when provided as a string
  private static readonly VALID_AFFINITIES = [
    "rbf",
    "nearest_neighbors",
    "precomputed",
  ] as const;

  constructor(params: SpectralClusteringParams) {
    // Freeze user params to avoid accidental mutation downstream.
    this.params = { ...params };

    SpectralClustering.validateParams(this.params);
  }

  /**
   * Fits the Spectral Clustering model to the input data and stores the
   * resulting cluster labels in {@link labels_}.
   *
   * Pipeline (Ng et al., 2001):
   *   1. Build similarity graph – affinity matrix A (task-9)
   *   2. Compute normalised Laplacian L = I − D^{-1/2} A D^{-1/2} (task-10)
   *   3. Obtain k smallest eigenvectors of L → embedding U (n × k) (task-10.1)
   *   4. Row-normalise U to unit length (spectral embedding)
   *   5. Run K-Means on the rows of U to obtain final labels (task-11)
   */
  async fit(_X: DataMatrix): Promise<void> {


    /* ---------------------------- 0) Input -------------------------------- */
    const Xtensor: tf.Tensor2D = _X instanceof tf.Tensor
      ? (tf.cast(_X as tf.Tensor2D, "float32") as tf.Tensor2D)
      : tf.tensor2d(_X as number[][], undefined, "float32");

    /* ---------------------------- 1) Affinity ----------------------------- */
    this.affinityMatrix_ = SpectralClustering.computeAffinityMatrix(
      Xtensor,
      this.params,
    );

    const affinitySum = (await this.affinityMatrix_.sum().data())[0];
    if (affinitySum === 0) {
      throw new Error(
        "Affinity matrix contains only zeros – cannot perform spectral clustering.",
      );
    }

    /* ---------------------------- 2) Laplacian ---------------------------- */
    const { normalised_laplacian } = await import("../utils/laplacian");
    const laplacian: tf.Tensor2D = tf.tidy(() =>
      normalised_laplacian(this.affinityMatrix_ as tf.Tensor2D),
    ) as tf.Tensor2D;

    /* --------------------- 3) Smallest eigenvectors ----------------------- */
    const { smallest_eigenvectors } = await import("../utils/laplacian");
    const U = smallest_eigenvectors(laplacian, this.params.nClusters);

    /* ------------------------ 4) Row normalise ---------------------------- */
    const eps = 1e-10;
    const U_norm: tf.Tensor2D = tf.tidy(() => {
      const rowNorm = U.norm("euclidean", 1).expandDims(1);
      return U.div(rowNorm.add(eps));
    }) as tf.Tensor2D;

    /* -------------------------- 5) K-Means -------------------------------- */
    const { KMeans } = await import("./kmeans");
    const km = new KMeans({
      nClusters: this.params.nClusters,
      randomState: this.params.randomState,
    });

    await km.fit(U_norm);

    this.labels_ = km.labels_;

    /* --------------------------- Clean-up --------------------------------- */
    laplacian.dispose();
    U.dispose();
    U_norm.dispose();

    if (!(_X instanceof tf.Tensor)) {
      Xtensor.dispose();
    }
  }

  async fitPredict(X: DataMatrix): Promise<LabelVector> {
    await this.fit(X);
    if (this.labels_ == null) {
      throw new Error("SpectralClustering failed to compute labels.");
    }
    return this.labels_;
  }

  /* ------------------------------------------------------------------- */
  /*                     Static parameter validation                       */
  /* ------------------------------------------------------------------- */

  private static validateParams(params: SpectralClusteringParams): void {
    const {
      nClusters,
      affinity = "rbf",
      gamma,
      nNeighbors,
    } = params;

    // nClusters must be a positive integer
    if (!Number.isInteger(nClusters) || nClusters < 1) {
      throw new Error("nClusters must be a positive integer (>= 1).");
    }

    // Affinity string or callable
    const isCallable = typeof affinity === "function";
    if (!isCallable && !SpectralClustering.VALID_AFFINITIES.includes(affinity)) {
      throw new Error(
        `Invalid affinity '${affinity}'. Must be one of ${SpectralClustering.VALID_AFFINITIES.join(", ")} or a callable.`,
      );
    }

    // gamma checks (only relevant for RBF affinity when provided as string)
    if (!isCallable && affinity === "rbf") {
      if (gamma !== undefined && (typeof gamma !== "number" || gamma <= 0)) {
        throw new Error("gamma must be a positive number if specified.");
      }
    } else if (gamma !== undefined) {
      // If affinity is not RBF but user supplied gamma, warn
      throw new Error("gamma is only applicable when affinity is 'rbf'.");
    }

    // nNeighbors checks for nearest_neighbors affinity
    if (!isCallable && affinity === "nearest_neighbors") {
      const k = nNeighbors ?? 10; // default will be set later
      if (!Number.isInteger(k) || k < 1) {
        throw new Error("nNeighbors must be a positive integer (>= 1).");
      }
    } else if (nNeighbors !== undefined) {
      throw new Error(
        "nNeighbors is only applicable when affinity is 'nearest_neighbors'.",
      );
    }

    // precomputed: gamma / nNeighbors not allowed
    if (!isCallable && affinity === "precomputed") {
      if (gamma !== undefined) {
        throw new Error("gamma is not applicable when affinity is 'precomputed'.");
      }
      if (nNeighbors !== undefined) {
        throw new Error("nNeighbors is not applicable when affinity is 'precomputed'.");
      }
    }
  }

  /* ------------------------------------------------------------------- */
  /*                       Affinity matrix utilities                       */
  /* ------------------------------------------------------------------- */

  private static computeAffinityMatrix(
    X: tf.Tensor2D,
    params: SpectralClusteringParams,
  ): tf.Tensor2D {
    const { affinity = "rbf" } = params;

    // -------------------------- Callable affinity ------------------------ //
    if (typeof affinity === "function") {
      const A = affinity(X);
      SpectralClustering.validateAffinityMatrix(A);
      return A;
    }

    // ---------------------------- Precomputed ---------------------------- //
    if (affinity === "precomputed") {
      SpectralClustering.validateAffinityMatrix(X);
      return X;
    }

    // Import on-demand to avoid circular deps when this file is imported by
    // the utils module (which is unlikely but safe).
    // eslint-disable-next-line @typescript-eslint/no-var-requires, node/no-missing-import
    const {
      compute_rbf_affinity,
      compute_knn_affinity,
    } = require("../utils/affinity") as typeof import("../utils/affinity");

    if (affinity === "rbf") {
      return compute_rbf_affinity(X, params.gamma);
    }

    // nearest_neighbors
    const k = params.nNeighbors ?? 10; // Default consistent with scikit-learn
    return compute_knn_affinity(X, k);
  }

  /**
   * Validates that the provided tensor is a proper affinity / similarity
   * matrix suitable for spectral clustering.
   *   • Must be 2-D & **square**
   *   • Must be **symmetric** (within tolerance)
   *   • Must be **non-negative** (entries ≥ 0)
   */
  private static validateAffinityMatrix(A: tf.Tensor2D): void {
    if (A.shape.length !== 2 || A.shape[0] !== A.shape[1]) {
      throw new Error("Affinity matrix must be square (n × n).");
    }

    const n = A.shape[0];

    // Check symmetry & non-negativity using small tolerances.
    tf.tidy(() => {
      const tol = 1e-6;
      const diff = A.sub(A.transpose()).abs();
      const maxDiff = diff.max().dataSync()[0];
      if (maxDiff > tol) {
        throw new Error("Affinity matrix must be symmetric.");
      }

      const minVal = A.min().dataSync()[0];
      if (minVal < -tol) {
        throw new Error("Affinity matrix must be non-negative.");
      }
    });
  }
}
