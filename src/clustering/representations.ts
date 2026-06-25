import * as tf from '../backend/adapter';

/** Lets downstream code work identically across estimators regardless of how each exposes cluster representatives. */
export interface ClusterRepresentations {
  centroids_?: tf.Tensor2D | null;
  /**
   * Library-defined: scikit-learn exposes no equivalent attribute.
   * `-1` if that cluster has no assigned samples; ties on distance resolve to the lowest sample index.
   */
  medoid_indices_?: Int32Array | null;
  /**
   * Library-defined: scikit-learn exposes no equivalent attribute.
   * Map key is cluster id; value is the exemplar sample index. Ties resolve to the lowest sample index.
   */
  exemplar_indices_?: Map<number, number> | null;
}
