import tf from '../tf-adapter';
import { SpectralClustering } from './spectral';
import {
  DataMatrix,
  LabelVector as _LabelVector,
  SpectralClusteringParams,
} from './types';

/**
 * SpectralClustering with consensus clustering to improve robustness.
 * Runs k-means multiple times and takes majority vote for each point.
 */
export class SpectralClusteringConsensus extends SpectralClustering {
  private consensusRuns: number;

  constructor(params: SpectralClusteringParams & { consensusRuns?: number }) {
    super(params);
    this.consensusRuns = params.consensusRuns ?? 50;
  }

  async fit(X: DataMatrix): Promise<void> {
    // We need a custom implementation that recomputes the embedding
    // because the parent class doesn't store it

    const Xtensor: tf.Tensor2D =
      X instanceof tf.Tensor
        ? (tf.cast(X as tf.Tensor2D, 'float32') as tf.Tensor2D)
        : tf.tensor2d(X as number[][], undefined, 'float32');

    // Build affinity matrix (reuse parent logic)
    const computeAffinityMatrix = (
      SpectralClustering as unknown as {
        computeAffinityMatrix: (
          X: tf.Tensor2D,
          params: SpectralClusteringParams,
        ) => tf.Tensor2D;
      }
    ).computeAffinityMatrix;
    this.affinityMatrix_ = computeAffinityMatrix(Xtensor, this.params);

    const affinitySum = (await this.affinityMatrix_!.sum().data())[0];
    if (affinitySum === 0) {
      throw new Error(
        'Affinity matrix contains only zeros – cannot perform spectral clustering.',
      );
    }

    // Detect connected components
    const { detectConnectedComponents } = await import(
      '../utils/connected_components'
    );
    const { numComponents, isFullyConnected, componentLabels } =
      detectConnectedComponents(this.affinityMatrix_! as tf.Tensor2D);

    if (!isFullyConnected) {
      console.warn(
        'Graph is not fully connected, spectral embedding may not work as expected.',
      );
    }

    let U: tf.Tensor2D;

    // If graph is disconnected and has enough components, use component indicators
    if (!isFullyConnected && numComponents >= this.params.nClusters) {
      const { createComponentIndicators } = await import(
        '../utils/component_indicators'
      );
      U = createComponentIndicators(
        componentLabels,
        numComponents,
        numComponents,
      );
    } else {
      // Standard approach: compute Laplacian and eigenvectors
      const { normalised_laplacian } = await import('../utils/laplacian');
      const laplacian = tf.tidy(() =>
        normalised_laplacian(this.affinityMatrix_! as tf.Tensor2D),
      );

      const { smallest_eigenvectors_with_values } = await import(
        '../utils/smallest_eigenvectors_with_values'
      );
      const numEigenvectors = Math.max(this.params.nClusters, numComponents);
      const { eigenvectors: U_full, eigenvalues } =
        smallest_eigenvectors_with_values(laplacian, numEigenvectors);

      // Apply diffusion map scaling
      const U_scaled = tf.tidy(() => {
        const numToUse = this.params.nClusters;
        const eigenvals = tf.slice(eigenvalues, [0], [numToUse]) as tf.Tensor1D;
        const scalingFactors = tf.sqrt(
          tf.maximum(tf.scalar(0), tf.sub(tf.scalar(1), eigenvals)),
        ) as tf.Tensor1D;
        const scalingFactors2D = scalingFactors.reshape([1, -1]) as tf.Tensor2D;
        const U_selected = tf.slice(
          U_full,
          [0, 0],
          [-1, numToUse],
        ) as tf.Tensor2D;
        return U_selected.mul(scalingFactors2D) as tf.Tensor2D;
      });

      U = U_scaled;
      laplacian.dispose();
      eigenvalues.dispose();
      U_full.dispose();
    }

    // Run k-means multiple times with different random seeds
    const { KMeans } = await import('./kmeans');
    const allLabels: number[][] = [];

    for (let run = 0; run < this.consensusRuns; run++) {
      const km = new KMeans({
        nClusters: this.params.nClusters,
        randomState: (this.params.randomState ?? 42) + run,
        nInit: 1, // Single init per run, we handle multiple runs here
      });

      await km.fit(U);
      allLabels.push(km.labels_ as number[]);
    }

    // Consensus: for each point, take the most common label
    const n = allLabels[0].length;
    const consensusLabels: number[] = [];

    for (let i = 0; i < n; i++) {
      // Get all labels for point i
      const labelsForPoint = allLabels.map((labels) => labels[i]);

      // Count occurrences
      const counts = new Map<number, number>();
      for (const label of labelsForPoint) {
        counts.set(label, (counts.get(label) || 0) + 1);
      }

      // Find most common label
      let maxCount = 0;
      let consensusLabel = 0;
      for (const [label, count] of counts) {
        if (count > maxCount) {
          maxCount = count;
          consensusLabel = label;
        }
      }

      consensusLabels.push(consensusLabel);
    }

    // Update labels
    this.labels_ = consensusLabels;

    // Cleanup
    U.dispose();
    if (!(X instanceof tf.Tensor)) {
      Xtensor.dispose();
    }
  }
}
