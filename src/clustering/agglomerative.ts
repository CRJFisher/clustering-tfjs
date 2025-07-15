import type { DataMatrix, LabelVector, AgglomerativeClusteringParams, BaseClustering } from "./types";

/**
 * Agglomerative (hierarchical) clustering estimator skeleton.
 *
 * Only the constructor, parameter validation and public property definitions
 * are implemented as part of this initial task. The actual clustering logic
 * will be added in subsequent tasks.
 */
export class AgglomerativeClustering
  implements BaseClustering<AgglomerativeClusteringParams>
{
  /**
   * Hyper-parameters describing the behaviour of this instance.
   */
  public readonly params: AgglomerativeClusteringParams;

  /**
   * Cluster labels produced by `fit` / `fitPredict`.
   *
   * Populated after calling `fit`.
   */
  public labels_: LabelVector | null = null;

  /**
   * Children of each non-leaf node in the hierarchical clustering tree.
   * Shape: `(nSamples-1, 2)` where each row gives the indices of the merged
   * clusters. Lazily populated by future implementation.
   */
  public children_: number[][] | null = null;

  /**
   * Number of leaves in the hierarchical clustering tree (equals `nSamples`).
   */
  public nLeaves_: number | null = null;

  /**
   * Allowed linkage strategies.
   */
  private static readonly VALID_LINKAGES = [
    "ward",
    "complete",
    "average",
    "single",
  ] as const;

  /**
   * Allowed distance metrics.
   */
  private static readonly VALID_METRICS = [
    "euclidean",
    "manhattan",
    "cosine",
  ] as const;

  constructor(params: AgglomerativeClusteringParams) {
    // Perform a shallow copy to freeze user input and avoid side effects.
    this.params = { ...params };

    AgglomerativeClustering.validateParams(this.params);
  }

  /**
   * Fits the estimator to the provided data matrix.
   *
   * Note: The actual algorithm is not implemented yet. The stub only exists so
   * the public interface is complete and unit tests can assert that the method
   * is callable.
   */
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async fit(_X: DataMatrix): Promise<void> {
    throw new Error("AgglomerativeClustering.fit is not implemented yet.");
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async fitPredict(_X: DataMatrix): Promise<LabelVector> {
    // A real implementation would call `fit` and return `labels_`. For now we
    // simply make it clear to callers that the functionality is missing.
    throw new Error("AgglomerativeClustering.fitPredict is not implemented yet.");
  }

  /* --------------------------------------------------------------------- */
  /*                         Parameter Validation                          */
  /* --------------------------------------------------------------------- */

  private static validateParams(params: AgglomerativeClusteringParams): void {
    const { nClusters, linkage = "ward", metric = "euclidean" } = params;

    // nClusters must be a positive integer
    if (!Number.isInteger(nClusters) || nClusters < 1) {
      throw new Error("nClusters must be a positive integer (>= 1).");
    }

    // linkage value
    if (!AgglomerativeClustering.VALID_LINKAGES.includes(linkage)) {
      throw new Error(
        `Invalid linkage '${linkage}'. Must be one of ${AgglomerativeClustering.VALID_LINKAGES.join(", ")}.`,
      );
    }

    // metric value
    if (!AgglomerativeClustering.VALID_METRICS.includes(metric)) {
      throw new Error(
        `Invalid metric '${metric}'. Must be one of ${AgglomerativeClustering.VALID_METRICS.join(", ")}.`,
      );
    }

    // Additional consistency check: Ward linkage requires Euclidean distance.
    if (linkage === "ward" && metric !== "euclidean") {
      throw new Error("Ward linkage requires metric to be 'euclidean'.");
    }
  }
}

