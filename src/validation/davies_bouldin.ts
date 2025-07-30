import * as tf from "@tensorflow/tfjs-node";
import { DataMatrix, LabelVector } from "../clustering/types";

/**
 * Computes the Davies-Bouldin score.
 * 
 * The Davies-Bouldin index is defined as the average similarity measure
 * of each cluster with its most similar cluster. Lower values indicate
 * better clustering (clusters are more separated).
 * 
 * Formula: DB = (1/k) * sum(max_{i≠j}(R_{ij}))
 * where R_{ij} = (s_i + s_j) / d_{ij}
 * - s_i = average distance from points in cluster i to its centroid
 * - d_{ij} = distance between centroids of clusters i and j
 * 
 * @param X - Data matrix of shape [n_samples, n_features]
 * @param labels - Cluster labels for each sample
 * @returns The Davies-Bouldin score (lower is better)
 * @throws Error if k <= 1
 */
export function daviesBouldin(
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
      throw new Error("Davies-Bouldin score requires at least 2 clusters");
    }
    
    // Compute centroids and intra-cluster dispersions
    const centroids: tf.Tensor1D[] = [];
    const dispersions: number[] = [];
    
    for (const label of uniqueLabels) {
      // Get indices for this cluster
      const clusterIndices: number[] = [];
      for (let i = 0; i < labelArray.length; i++) {
        if (labelArray[i] === label) {
          clusterIndices.push(i);
        }
      }
      
      const clusterSize = clusterIndices.length;
      if (clusterSize === 0) continue;
      
      // Extract cluster points
      const clusterData = tf.gather(data, clusterIndices);
      
      // Compute centroid
      const centroid = clusterData.mean(0) as tf.Tensor1D;
      centroids.push(centroid);
      
      // Compute intra-cluster dispersion (average distance to centroid)
      if (clusterSize > 1) {
        const diff = clusterData.sub(centroid.reshape([1, -1]));
        const distances = tf.sqrt(diff.square().sum(1));
        const avgDistance = distances.mean().dataSync()[0];
        dispersions.push(avgDistance);
        distances.dispose();
        diff.dispose();
      } else {
        // Single point cluster has zero dispersion
        dispersions.push(0);
      }
      
      // Clean up
      clusterData.dispose();
    }
    
    // Compute inter-cluster distances and similarity ratios
    const maxSimilarities: number[] = [];
    
    for (let i = 0; i < k; i++) {
      let maxSimilarity = 0;
      
      for (let j = 0; j < k; j++) {
        if (i === j) continue;
        
        // Compute distance between centroids
        const diff = centroids[i].sub(centroids[j]);
        const distance = tf.sqrt(diff.square().sum()).dataSync()[0];
        diff.dispose();
        
        // Avoid division by zero
        if (distance === 0) {
          // If centroids are identical, set similarity to infinity
          maxSimilarity = Infinity;
          break;
        }
        
        // Compute similarity ratio R_ij = (s_i + s_j) / d_ij
        const similarity = (dispersions[i] + dispersions[j]) / distance;
        
        if (similarity > maxSimilarity) {
          maxSimilarity = similarity;
        }
      }
      
      maxSimilarities.push(maxSimilarity);
    }
    
    // Clean up centroids
    for (const centroid of centroids) {
      centroid.dispose();
    }
    
    // Compute Davies-Bouldin index as average of maximum similarities
    const dbScore = maxSimilarities.reduce((sum, val) => sum + val, 0) / k;
    
    return dbScore;
  });
}

/**
 * Computes the Davies-Bouldin score with optimized memory usage.
 * This version minimizes tensor allocations and disposals.
 * 
 * @param X - Data matrix of shape [n_samples, n_features]
 * @param labels - Cluster labels for each sample
 * @returns The Davies-Bouldin score (lower is better)
 */
export function daviesBouldinEfficient(
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
    throw new Error("Davies-Bouldin score requires at least 2 clusters");
  }
  
  // Store centroids and dispersions
  const centroidArrays: number[][] = [];
  const dispersions: number[] = [];
  
  // Compute centroids and dispersions
  for (const label of uniqueLabels) {
    const clusterIndices = labelArray
      .map((l, i) => l === label ? i : -1)
      .filter(i => i >= 0);
    
    if (clusterIndices.length === 0) continue;
    
    tf.tidy(() => {
      const clusterData = tf.gather(data, clusterIndices);
      const centroid = clusterData.mean(0) as tf.Tensor1D;
      centroidArrays.push(Array.from(centroid.dataSync()));
      
      if (clusterIndices.length > 1) {
        const diff = clusterData.sub(centroid.reshape([1, -1]));
        const distances = tf.sqrt(diff.square().sum(1));
        dispersions.push(distances.mean().dataSync()[0]);
      } else {
        dispersions.push(0);
      }
    });
  }
  
  // Clean up data tensor if we created it
  if (!(X instanceof tf.Tensor)) {
    data.dispose();
  }
  
  // Compute Davies-Bouldin index
  let dbSum = 0;
  
  for (let i = 0; i < k; i++) {
    let maxSimilarity = 0;
    
    for (let j = 0; j < k; j++) {
      if (i === j) continue;
      
      // Compute Euclidean distance between centroids
      let distance = 0;
      for (let d = 0; d < centroidArrays[i].length; d++) {
        const diff = centroidArrays[i][d] - centroidArrays[j][d];
        distance += diff * diff;
      }
      distance = Math.sqrt(distance);
      
      if (distance === 0) {
        maxSimilarity = Infinity;
        break;
      }
      
      const similarity = (dispersions[i] + dispersions[j]) / distance;
      if (similarity > maxSimilarity) {
        maxSimilarity = similarity;
      }
    }
    
    dbSum += maxSimilarity;
  }
  
  return dbSum / k;
}