import * as tf from '../tf-adapter';
import { DataMatrix, LabelVector } from '../clustering/types';
import { isTensor } from './tensor-utils';

/**
 * Computes the Within-Cluster Sum of Squares (WSS / inertia) for a clustering.
 *
 * WSS is the sum of squared Euclidean distances from each point to its
 * cluster centroid. Lower WSS indicates tighter clusters.
 *
 * @param X - Data matrix of shape [n_samples, n_features]
 * @param labels - Cluster labels for each sample
 * @returns The within-cluster sum of squares
 */
export function computeWss(X: DataMatrix, labels: LabelVector): number {
  const data = isTensor(X) ? X : tf.tensor2d(X as number[][]);
  const labelArray = isTensor(labels)
    ? Array.from(labels.dataSync() as Float32Array).map((l) => Math.round(l))
    : (labels as number[]);

  const uniqueLabels = Array.from(new Set(labelArray));
  let wss = 0;

  for (const label of uniqueLabels) {
    tf.tidy(() => {
      const clusterIndices = labelArray
        .map((l, i) => (l === label ? i : -1))
        .filter((i) => i >= 0);

      if (clusterIndices.length === 0) return;

      const clusterData = tf.gather(data, clusterIndices);
      const centroid = clusterData.mean(0) as tf.Tensor1D;
      const diff = clusterData.sub(centroid.reshape([1, -1]));
      wss += diff.square().sum().dataSync()[0];
    });
  }

  if (!isTensor(X)) {
    (data as tf.Tensor2D).dispose();
  }

  return wss;
}
