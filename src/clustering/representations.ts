import * as tf from '../backend/adapter';

/**
 * Uniform surface for obtaining a representative vector per cluster, regardless
 * of which estimator produced the labels.
 *
 * Different algorithms expose different kinds of representatives:
 *
 * - **KMeans** computes synthetic `centroids_` (cluster means).
 * - **AgglomerativeClustering** and **SpectralClustering** produce only labels,
 *   so they expose `medoid_indices_` — the index of the real sample closest to
 *   each cluster's mean.
 * - **HDBSCAN** exposes `exemplar_indices_` — the most-persistent point per
 *   cluster.
 *
 * Implementing this single contract lets downstream code (summarization,
 * labelling, nearest-representative lookups) work identically across estimators.
 */
export interface ClusterRepresentations {
  /** Synthetic cluster centres (KMeans), shape `n_clusters × n_features`. */
  centroids_?: tf.Tensor2D | null;
  /**
   * Index of the representative sample per cluster (Agglomerative, Spectral).
   * Position `c` holds the medoid index of cluster `c`, or `-1` if that cluster
   * has no assigned samples.
   *
   * Library-defined: scikit-learn exposes no equivalent attribute. The medoid
   * is the in-cluster sample closest to the cluster mean under the fit metric;
   * ties resolve towards the lowest sample index.
   */
  medoid_indices_?: Int32Array | null;
  /**
   * Exemplar sample index per cluster id (HDBSCAN).
   *
   * Library-defined: scikit-learn exposes no equivalent attribute. The
   * exemplar is the most-persistent (highest-λ) point of each cluster; ties
   * resolve towards the lowest sample index.
   */
  exemplar_indices_?: Map<number, number> | null;
}
