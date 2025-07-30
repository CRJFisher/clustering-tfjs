import * as tf from "@tensorflow/tfjs-node";
import { DataMatrix, LabelVector } from "../clustering/types";

/**
 * Computes the Calinski-Harabasz score (also known as Variance Ratio Criterion).
 * 
 * The score is defined as the ratio of the between-cluster dispersion to the
 * within-cluster dispersion. Higher values indicate better-defined clusters.
 * 
 * Formula: CH = (BSS / (k - 1)) / (WSS / (n - k))
 * where:
 * - BSS = between-cluster sum of squares
 * - WSS = within-cluster sum of squares
 * - k = number of clusters
 * - n = number of samples
 * 
 * @param X - Data matrix of shape [n_samples, n_features]
 * @param labels - Cluster labels for each sample
 * @returns The Calinski-Harabasz score (higher is better)
 * @throws Error if k <= 1 or k >= n_samples
 */
export function calinskiHarabasz(
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
    const nFeatures = data.shape[1];
    
    // Get unique labels and count
    const uniqueLabels = Array.from(new Set(labelArray));
    const k = uniqueLabels.length;
    
    // Validate inputs
    if (k <= 1) {
      throw new Error("Calinski-Harabasz score requires at least 2 clusters");
    }
    if (k >= n) {
      throw new Error("Number of clusters must be less than number of samples");
    }
    
    // Compute global centroid
    const globalCentroid = data.mean(0) as tf.Tensor1D;
    
    // Initialize accumulators
    let withinClusterSS = 0;
    let betweenClusterSS = 0;
    
    // Process each cluster
    for (const label of uniqueLabels) {
      // Get indices for this cluster
      const clusterIndices: number[] = [];
      for (let i = 0; i < labelArray.length; i++) {
        if (labelArray[i] === label) {
          clusterIndices.push(i);
        }
      }
      
      const clusterSize = clusterIndices.length;
      
      // Extract cluster points
      const clusterData = tf.gather(data, clusterIndices);
      
      // Compute cluster centroid
      const clusterCentroid = clusterData.mean(0) as tf.Tensor1D;
      
      // Within-cluster sum of squares
      const diff = clusterData.sub(clusterCentroid.reshape([1, -1]));
      const squaredDiff = diff.square();
      withinClusterSS += squaredDiff.sum().dataSync()[0];
      
      // Between-cluster sum of squares
      const centroidDiff = clusterCentroid.sub(globalCentroid);
      const centroidDiffSquared = centroidDiff.square().sum();
      betweenClusterSS += clusterSize * centroidDiffSquared.dataSync()[0];
      
      // Clean up tensors
      clusterData.dispose();
      clusterCentroid.dispose();
      diff.dispose();
      squaredDiff.dispose();
      centroidDiff.dispose();
      centroidDiffSquared.dispose();
    }
    
    // Compute Calinski-Harabasz score
    const score = (betweenClusterSS / (k - 1)) / (withinClusterSS / (n - k));
    
    // Clean up
    globalCentroid.dispose();
    
    return score;
  });
}

/**
 * Computes the Calinski-Harabasz score in a memory-efficient manner for large datasets.
 * This version processes clusters sequentially to minimize memory usage.
 * 
 * @param X - Data matrix of shape [n_samples, n_features]
 * @param labels - Cluster labels for each sample
 * @returns The Calinski-Harabasz score
 */
export function calinskiHarabaszEfficient(
  X: DataMatrix,
  labels: LabelVector
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
    throw new Error("Calinski-Harabasz score requires at least 2 clusters");
  }
  if (k >= n) {
    if (!(X instanceof tf.Tensor)) {
      data.dispose();
    }
    throw new Error("Number of clusters must be less than number of samples");
  }
  
  // Compute global centroid
  const globalCentroid = tf.tidy(() => data.mean(0) as tf.Tensor1D);
  
  let withinClusterSS = 0;
  let betweenClusterSS = 0;
  
  // Process each cluster
  for (const label of uniqueLabels) {
    tf.tidy(() => {
      // Get cluster indices
      const clusterIndices = labelArray
        .map((l, i) => l === label ? i : -1)
        .filter(i => i >= 0);
      
      const clusterSize = clusterIndices.length;
      
      // Extract cluster data
      const clusterData = tf.gather(data, clusterIndices);
      const clusterCentroid = clusterData.mean(0) as tf.Tensor1D;
      
      // Within-cluster SS
      const diff = clusterData.sub(clusterCentroid.reshape([1, -1]));
      withinClusterSS += diff.square().sum().dataSync()[0];
      
      // Between-cluster SS
      const centroidDiff = clusterCentroid.sub(globalCentroid);
      betweenClusterSS += clusterSize * centroidDiff.square().sum().dataSync()[0];
    });
  }
  
  // Clean up
  globalCentroid.dispose();
  if (!(X instanceof tf.Tensor)) {
    data.dispose();
  }
  
  // Compute score
  return (betweenClusterSS / (k - 1)) / (withinClusterSS / (n - k));
}