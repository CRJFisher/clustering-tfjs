import * as tf from '../backend/adapter';
import { SpectralClustering } from './spectral';
import {
  DataMatrix,
  SpectralClusteringParams,
} from './types';
import { is_tensor } from '../tensor/tensor_guards';

/**
 * SpectralClustering with consensus clustering to improve robustness.
 * Runs k-means multiple times and takes majority vote for each point.
 */
export class SpectralClusteringConsensus extends SpectralClustering {
  private consensus_runs: number;

  constructor(params: SpectralClusteringParams & { consensus_runs?: number }) {
    super(params);
    this.consensus_runs = params.consensus_runs ?? 50;
  }

  async fit(X: DataMatrix): Promise<void> {
    // We need a custom implementation that recomputes the embedding
    // because the parent class doesn't store it

    const Xtensor: tf.Tensor2D =
      is_tensor(X)
        ? (tf.cast(X as tf.Tensor2D, 'float32') as tf.Tensor2D)
        : tf.tensor2d(X as number[][], undefined, 'float32');

    // Build affinity matrix (reuse parent logic)
    const compute_affinity_matrix = (
      SpectralClustering as unknown as {
        compute_affinity_matrix: (
          X: tf.Tensor2D,
          params: SpectralClusteringParams,
        ) => tf.Tensor2D;
      }
    ).compute_affinity_matrix;
    this.affinity_matrix_ = compute_affinity_matrix(Xtensor, this.params);

    const sum_tensor = this.affinity_matrix_!.sum();
    const affinity_sum = (await sum_tensor.data())[0];
    sum_tensor.dispose();
    if (affinity_sum === 0) {
      throw new Error(
        'Affinity matrix contains only zeros – cannot perform spectral clustering.',
      );
    }

    // Detect connected components
    const { detect_connected_components } = await import(
      '../graph/connected_components'
    );
    const { num_components, is_fully_connected, component_labels } =
      detect_connected_components(this.affinity_matrix_! as tf.Tensor2D);

    if (!is_fully_connected) {
      console.warn(
        'Graph is not fully connected, spectral embedding may not work as expected.',
      );
    }

    let U: tf.Tensor2D;

    // If graph is disconnected and has enough components, use component indicators
    if (!is_fully_connected && num_components >= this.params.n_clusters) {
      const { create_component_indicators } = await import(
        '../graph/component_indicators'
      );
      U = create_component_indicators(
        component_labels,
        num_components,
        num_components,
      );
    } else {
      // Standard approach: compute Laplacian and eigenvectors
      // Must pass return_diag=true to get sqrt_degrees for normalization
      const { normalised_laplacian } = await import('../graph/laplacian');
      const { laplacian, sqrt_degrees } = tf.tidy(() =>
        normalised_laplacian(this.affinity_matrix_! as tf.Tensor2D, true),
      );

      const { smallest_eigenvectors_with_values } = await import(
        '../eigen/smallest_eigenvectors_with_values'
      );
      const num_eigenvectors = Math.max(this.params.n_clusters, num_components);
      const { eigenvectors: U_full, eigenvalues } =
        smallest_eigenvectors_with_values(laplacian, num_eigenvectors);

      // Apply sklearn's normalization: divide by D^{1/2}
      const U_scaled = tf.tidy(() => {
        const num_to_use = this.params.n_clusters;
        const U_selected = tf.slice(
          U_full,
          [0, 0],
          [-1, num_to_use],
        ) as tf.Tensor2D;
        // sqrt_degrees is D^{-1/2}, so D^{1/2} = pow(sqrt_degrees, -1)
        const sqrt_deg = tf.pow(sqrt_degrees, -1) as tf.Tensor1D;
        const sqrt_deg_col = sqrt_deg.reshape([-1, 1]) as tf.Tensor2D;
        return U_selected.div(sqrt_deg_col) as tf.Tensor2D;
      });

      U = U_scaled;
      laplacian.dispose();
      sqrt_degrees.dispose();
      eigenvalues.dispose();
      U_full.dispose();
    }

    // Run k-means multiple times with different random seeds
    const { KMeans } = await import('./kmeans');
    const all_labels: number[][] = [];

    for (let run = 0; run < this.consensus_runs; run++) {
      const km = new KMeans({
        n_clusters: this.params.n_clusters,
        random_state: (this.params.random_state ?? 42) + run,
        n_init: 1, // Single init per run, we handle multiple runs here
      });

      await km.fit(U);
      all_labels.push(km.labels_!);
      km.dispose();
    }

    // Consensus: for each point, take the most common label
    const n = all_labels[0].length;
    const consensus_labels: number[] = [];

    for (let i = 0; i < n; i++) {
      // Get all labels for point i
      const labels_for_point = all_labels.map((labels) => labels[i]);

      // Count occurrences
      const counts = new Map<number, number>();
      for (const label of labels_for_point) {
        counts.set(label, (counts.get(label) || 0) + 1);
      }

      // Find most common label
      let max_count = 0;
      let consensus_label = 0;
      for (const [label, count] of counts) {
        if (count > max_count) {
          max_count = count;
          consensus_label = label;
        }
      }

      consensus_labels.push(consensus_label);
    }

    // Update labels
    this.labels_ = consensus_labels;

    // Cleanup
    U.dispose();
    Xtensor.dispose();
  }
}
