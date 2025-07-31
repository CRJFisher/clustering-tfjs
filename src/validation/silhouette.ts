import * as tf from "../utils/tensorflow";
import { DataMatrix, LabelVector } from "../clustering/types";

/**
 * Computes the Silhouette score.
 * 
 * The silhouette coefficient for a sample is (b - a) / max(a, b) where:
 * - a is the mean distance between a sample and all other points in the same cluster
 * - b is the mean distance between a sample and all points in the nearest cluster
 * 
 * The score ranges from -1 to +1:
 * - +1: Sample is far from neighboring clusters (well clustered)
 * - 0: Sample is on or very close to the decision boundary
 * - -1: Sample might have been assigned to the wrong cluster
 * 
 * @param X - Data matrix of shape [n_samples, n_features]
 * @param labels - Cluster labels for each sample
 * @returns The mean silhouette score across all samples
 * @throws Error if k <= 1
 */
export function silhouetteScore(
  X: DataMatrix,
  labels: LabelVector
): number {
  return tf.tidy(() => {
    // Convert inputs to tensors
    const data = X instanceof tf.Tensor ? X as tf.Tensor2D : tf.tensor2d(X as number[][]);
    const labelArray = labels instanceof tf.Tensor 
      ? Array.from(labels.dataSync() as Float32Array).map(l => Math.round(l))
      : labels as number[];
    
    const n = data.shape[0];
    
    // Get unique labels
    const uniqueLabels = Array.from(new Set(labelArray));
    const k = uniqueLabels.length;
    
    // Validate inputs
    if (k <= 1) {
      throw new Error("Silhouette score requires at least 2 clusters");
    }
    
    // Compute pairwise distances
    // D[i,j] = ||x_i - x_j||^2
    const xNorm = data.square().sum(1).reshape([n, 1]);
    const xNormT = xNorm.reshape([1, n]);
    const cross = tf.matMul(data, data.transpose());
    const distances = tf.sqrt(
      tf.maximum(
        tf.scalar(0),
        xNorm.add(xNormT).sub(cross.mul(2))
      )
    ) as tf.Tensor2D;
    
    // Compute silhouette for each sample
    const silhouetteValues: number[] = [];
    const distancesArray = distances.arraySync();
    
    for (let i = 0; i < n; i++) {
      const sampleLabel = labelArray[i];
      
      // Find indices of samples in same cluster and other clusters
      const sameClusterIndices: number[] = [];
      const otherClusterIndices: Map<number, number[]> = new Map();
      
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        
        if (labelArray[j] === sampleLabel) {
          sameClusterIndices.push(j);
        } else {
          const label = labelArray[j];
          if (!otherClusterIndices.has(label)) {
            otherClusterIndices.set(label, []);
          }
          otherClusterIndices.get(label)!.push(j);
        }
      }
      
      // Compute a(i): mean intra-cluster distance
      let a = 0;
      if (sameClusterIndices.length > 0) {
        for (const j of sameClusterIndices) {
          a += distancesArray[i][j];
        }
        a /= sameClusterIndices.length;
      }
      
      // Compute b(i): mean distance to nearest cluster
      let b = Infinity;
      for (const [_label, indices] of otherClusterIndices) {
        let meanDist = 0;
        for (const j of indices) {
          meanDist += distancesArray[i][j];
        }
        meanDist /= indices.length;
        
        if (meanDist < b) {
          b = meanDist;
        }
      }
      
      // Compute silhouette coefficient
      if (sameClusterIndices.length === 0) {
        // Single point in cluster
        silhouetteValues.push(0);
      } else {
        const s = (b - a) / Math.max(a, b);
        silhouetteValues.push(s);
      }
    }
    
    // Clean up
    xNorm.dispose();
    xNormT.dispose();
    cross.dispose();
    distances.dispose();
    
    // Return mean silhouette score
    return silhouetteValues.reduce((sum, val) => sum + val, 0) / n;
  });
}

/**
 * Computes the Silhouette score for specific samples (subset).
 * Useful for large datasets where computing all pairwise distances is prohibitive.
 * 
 * @param X - Data matrix of shape [n_samples, n_features]
 * @param labels - Cluster labels for each sample
 * @param sampleIndices - Indices of samples to compute silhouette for
 * @returns The mean silhouette score for the specified samples
 */
export function silhouetteScoreSubset(
  X: DataMatrix,
  labels: LabelVector,
  sampleIndices: number[]
): number {
  // Convert inputs
  const data = X instanceof tf.Tensor ? X as tf.Tensor2D : tf.tensor2d(X as number[][]);
  const labelArray = labels instanceof tf.Tensor 
    ? Array.from(labels.dataSync() as Float32Array).map(l => Math.round(l))
    : labels as number[];
  
  const n = data.shape[0];
  
  // Get unique labels
  const uniqueLabels = Array.from(new Set(labelArray));
  const k = uniqueLabels.length;
  
  // Validate
  if (k <= 1) {
    if (!(X instanceof tf.Tensor)) {
      data.dispose();
    }
    throw new Error("Silhouette score requires at least 2 clusters");
  }
  
  const silhouetteValues: number[] = [];
  
  // Process each sample in the subset
  for (const i of sampleIndices) {
    const sampleLabel = labelArray[i];
    
    // Get the sample point
    const samplePoint = tf.tidy(() => data.gather([i]));
    
    // Compute distances to all other points
    const distances = tf.tidy(() => {
      const diff = data.sub(samplePoint);
      return tf.sqrt(diff.square().sum(1)) as tf.Tensor1D;
    });
    
    const distArray = distances.dataSync() as Float32Array;
    
    // Compute a(i) and b(i)
    let a = 0;
    let aCount = 0;
    const clusterDistances: Map<number, { sum: number; count: number }> = new Map();
    
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      
      const dist = distArray[j];
      const label = labelArray[j];
      
      if (label === sampleLabel) {
        // Same cluster
        a += dist;
        aCount++;
      } else {
        // Other cluster
        if (!clusterDistances.has(label)) {
          clusterDistances.set(label, { sum: 0, count: 0 });
        }
        const cluster = clusterDistances.get(label)!;
        cluster.sum += dist;
        cluster.count++;
      }
    }
    
    // Mean intra-cluster distance
    if (aCount > 0) {
      a /= aCount;
    }
    
    // Find nearest cluster
    let b = Infinity;
    for (const [_label, { sum, count }] of clusterDistances) {
      const meanDist = sum / count;
      if (meanDist < b) {
        b = meanDist;
      }
    }
    
    // Compute silhouette coefficient
    if (aCount === 0) {
      // Single point in cluster
      silhouetteValues.push(0);
    } else {
      const s = (b - a) / Math.max(a, b);
      silhouetteValues.push(s);
    }
    
    // Clean up
    samplePoint.dispose();
    distances.dispose();
  }
  
  // Clean up data tensor if we created it
  if (!(X instanceof tf.Tensor)) {
    data.dispose();
  }
  
  // Return mean silhouette score
  return silhouetteValues.reduce((sum, val) => sum + val, 0) / silhouetteValues.length;
}