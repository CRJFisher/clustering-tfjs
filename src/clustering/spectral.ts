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
  ] as const;

  constructor(params: SpectralClusteringParams) {
    // Freeze user params to avoid accidental mutation downstream.
    this.params = { ...params };

    SpectralClustering.validateParams(this.params);
  }

  /** Fit the model to X (stub – no real computation yet). */
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async fit(_X: DataMatrix): Promise<void> {
    // In task-9 we implement the first step of Spectral Clustering: building
    // the affinity (similarity) matrix according to the configured metric.

    // Convert input to a 2-D float32 tensor if necessary.
    const Xtensor =
      _X instanceof tf.Tensor ? (_X as tf.Tensor2D) : tf.tensor2d(_X, undefined, "float32");

    // Compute affinity matrix and cache the result
    this.affinityMatrix_ = SpectralClustering.computeAffinityMatrix(
      Xtensor,
      this.params,
    );

    // Labels will be produced in later tasks – set placeholder null.
    this.labels_ = null;
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
  }

  /* ------------------------------------------------------------------- */
  /*                       Affinity matrix utilities                       */
  /* ------------------------------------------------------------------- */

  private static computeAffinityMatrix(
    X: tf.Tensor2D,
    params: SpectralClusteringParams,
  ): tf.Tensor2D {
    const { affinity = "rbf" } = params;

    if (typeof affinity === "function") {
      return affinity(X);
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
}
