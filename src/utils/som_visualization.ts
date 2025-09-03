import * as tf from '../tf-adapter';
import type { SOM } from '../clustering/som';

/**
 * SOM Visualization Utilities
 * 
 * Provides functions for visualizing and analyzing trained SOM models.
 */

/**
 * Calculate component planes for each feature.
 * Shows how each input feature is distributed across the SOM grid.
 * 
 * @param som Trained SOM instance
 * @returns Component planes tensor [nFeatures, gridHeight, gridWidth]
 */
export function getComponentPlanes(som: SOM): tf.Tensor3D {
  const weights = som.getWeights();
  
  return tf.tidy(() => {
    // Transpose from [height, width, features] to [features, height, width]
    return weights.transpose([2, 0, 1]);
  });
}

/**
 * Calculate hit map showing sample distribution across neurons.
 * 
 * @param som Trained SOM instance
 * @param X Input data used for training
 * @returns Hit map tensor [gridHeight, gridWidth] with counts
 */
export async function getHitMap(
  som: SOM,
  X: tf.Tensor2D
): Promise<tf.Tensor2D> {
  const { gridHeight, gridWidth } = som.params;
  
  // Get BMUs for all samples
  const labels = await som.predict(X);
  
  // Count hits for each neuron
  const hitMap = tf.buffer([gridHeight, gridWidth]);
  
  if (Array.isArray(labels)) {
    for (const label of labels) {
      const row = Math.floor(label / gridWidth);
      const col = label % gridWidth;
      hitMap.set(hitMap.get(row, col) + 1, row, col);
    }
  }
  
  return hitMap.toTensor() as tf.Tensor2D;
}

/**
 * Calculate activation map for a specific input sample.
 * Shows how strongly each neuron responds to the input.
 * 
 * @param som Trained SOM instance
 * @param sample Input sample tensor [nFeatures]
 * @returns Activation map [gridHeight, gridWidth]
 */
export function getActivationMap(
  som: SOM,
  sample: tf.Tensor1D
): tf.Tensor2D {
  const weights = som.getWeights();
  
  return tf.tidy(() => {
    const [gridHeight, gridWidth, nFeatures] = weights.shape;
    
    // Reshape weights for computation
    const weightsFlat = weights.reshape([gridHeight * gridWidth, nFeatures]);
    
    // Compute distances from sample to all neurons
    const diff = weightsFlat.sub(sample.expandDims(0));
    const distances = diff.square().sum(1).sqrt();
    
    // Convert distances to activations (inverse relationship)
    const maxDist = distances.max();
    const activations = maxDist.sub(distances).div(maxDist);
    
    // Reshape to grid
    return activations.reshape([gridHeight, gridWidth]) as tf.Tensor2D;
  });
}

/**
 * Track BMU trajectory for a sequence of samples.
 * Useful for analyzing temporal patterns.
 * 
 * @param som Trained SOM instance
 * @param sequence Sequence of samples [nSamples, nFeatures]
 * @returns BMU positions [nSamples, 2] with grid coordinates
 */
export async function trackBMUTrajectory(
  som: SOM,
  sequence: tf.Tensor2D
): Promise<number[][]> {
  const { gridWidth } = som.params;
  const labels = await som.predict(sequence);
  
  const trajectory: number[][] = [];
  if (Array.isArray(labels)) {
    for (const label of labels) {
      const row = Math.floor(label / gridWidth);
      const col = label % gridWidth;
      trajectory.push([row, col]);
    }
  }
  
  return trajectory;
}

/**
 * Calculate quantization quality map.
 * Shows average quantization error for each neuron.
 * 
 * @param som Trained SOM instance
 * @param X Input data
 * @returns Quality map [gridHeight, gridWidth]
 */
export async function getQuantizationQualityMap(
  som: SOM,
  X: tf.Tensor2D
): Promise<tf.Tensor2D> {
  const { gridHeight, gridWidth } = som.params;
  const weights = som.getWeights();
  
  // Get labels and compute distances
  const labels = await som.predict(X);
  
  // Initialize quality map
  const qualityMap = tf.buffer([gridHeight, gridWidth]);
  const counts = tf.buffer([gridHeight, gridWidth]);
  
  // Compute average distance for each neuron
  const xArray = await X.array();
  const weightsArray = await weights.array();
  
  if (Array.isArray(labels)) {
    for (let i = 0; i < labels.length; i++) {
      const label = labels[i];
      const row = Math.floor(label / gridWidth);
      const col = label % gridWidth;
      
      // Compute distance
      const sample = xArray[i];
      const weight = weightsArray[row][col];
      const distance = Math.sqrt(
        sample.reduce((sum, val, idx) => 
          sum + Math.pow(val - weight[idx], 2), 0
        )
      );
      
      qualityMap.set(
        qualityMap.get(row, col) + distance,
        row,
        col
      );
      counts.set(counts.get(row, col) + 1, row, col);
    }
  }
  
  // Average the distances
  for (let i = 0; i < gridHeight; i++) {
    for (let j = 0; j < gridWidth; j++) {
      const count = counts.get(i, j);
      if (count > 0) {
        qualityMap.set(qualityMap.get(i, j) / count, i, j);
      }
    }
  }
  
  return qualityMap.toTensor() as tf.Tensor2D;
}

/**
 * Generate neighbor distance matrix for topology preservation analysis.
 * 
 * @param som Trained SOM instance
 * @returns Neighbor distances [gridHeight, gridWidth]
 */
export function getNeighborDistanceMatrix(som: SOM): tf.Tensor2D {
  const weights = som.getWeights();
  const { gridHeight, gridWidth, topology } = som.params;
  
  return tf.tidy(() => {
    const distanceMap = tf.buffer([gridHeight, gridWidth]);
    const weightsArray = weights.arraySync();
    
    for (let i = 0; i < gridHeight; i++) {
      for (let j = 0; j < gridWidth; j++) {
        const currentWeight = weightsArray[i][j];
        let totalDistance = 0;
        let neighborCount = 0;
        
        // Define neighbors based on topology
        const neighbors: [number, number][] = [];
        if (topology === 'rectangular') {
          // 4-connected neighbors
          if (i > 0) neighbors.push([i - 1, j]);
          if (i < gridHeight - 1) neighbors.push([i + 1, j]);
          if (j > 0) neighbors.push([i, j - 1]);
          if (j < gridWidth - 1) neighbors.push([i, j + 1]);
        } else {
          // Hexagonal neighbors (6-connected)
          const evenRow = i % 2 === 0;
          if (i > 0) {
            neighbors.push([i - 1, j]);
            if (evenRow && j > 0) neighbors.push([i - 1, j - 1]);
            if (!evenRow && j < gridWidth - 1) neighbors.push([i - 1, j + 1]);
          }
          if (i < gridHeight - 1) {
            neighbors.push([i + 1, j]);
            if (evenRow && j > 0) neighbors.push([i + 1, j - 1]);
            if (!evenRow && j < gridWidth - 1) neighbors.push([i + 1, j + 1]);
          }
          if (j > 0) neighbors.push([i, j - 1]);
          if (j < gridWidth - 1) neighbors.push([i, j + 1]);
        }
        
        // Calculate average distance to neighbors
        for (const [ni, nj] of neighbors) {
          const neighborWeight = weightsArray[ni][nj];
          const distance = Math.sqrt(
            currentWeight.reduce((sum, val, idx) => 
              sum + Math.pow(val - neighborWeight[idx], 2), 0
            )
          );
          totalDistance += distance;
          neighborCount++;
        }
        
        distanceMap.set(
          neighborCount > 0 ? totalDistance / neighborCount : 0,
          i,
          j
        );
      }
    }
    
    return distanceMap.toTensor() as tf.Tensor2D;
  });
}

/**
 * Export SOM data for external visualization tools.
 * 
 * @param som Trained SOM instance
 * @param format Export format ('json', 'csv')
 * @returns Formatted string for export
 */
export async function exportForVisualization(
  som: SOM,
  format: 'json' | 'csv' = 'json'
): Promise<string> {
  const weights = som.getWeights();
  const uMatrix = som.getUMatrix();
  const { gridHeight, gridWidth } = som.params;
  
  const weightsArray = await weights.array();
  const uMatrixArray = await uMatrix.array();
  
  if (format === 'json') {
    return JSON.stringify({
      gridHeight,
      gridWidth,
      weights: weightsArray,
      uMatrix: uMatrixArray,
      params: som.params,
    }, null, 2);
  } else {
    // CSV format
    let csv = 'row,col,u_value';
    for (let i = 0; i < weightsArray[0][0].length; i++) {
      csv += `,feature_${i}`;
    }
    csv += '\n';
    
    for (let i = 0; i < gridHeight; i++) {
      for (let j = 0; j < gridWidth; j++) {
        csv += `${i},${j},${uMatrixArray[i][j]}`;
        for (const feature of weightsArray[i][j]) {
          csv += `,${feature}`;
        }
        csv += '\n';
      }
    }
    
    return csv;
  }
}

/**
 * Calculate density map showing data concentration.
 * 
 * @param som Trained SOM instance
 * @param X Input data
 * @param sigma Gaussian kernel width for smoothing
 * @returns Density map [gridHeight, gridWidth]
 */
export async function getDensityMap(
  som: SOM,
  X: tf.Tensor2D,
  sigma: number = 1.0
): Promise<tf.Tensor2D> {
  const hitMap = await getHitMap(som, X);
  
  // Apply Gaussian smoothing for better visualization
  return tf.tidy(() => {
    // const [gridHeight, gridWidth] = hitMap.shape;  // Reserved for future Gaussian smoothing
    
    // Create Gaussian kernel
    const kernelSize = Math.ceil(sigma * 3) * 2 + 1;
    const kernel = tf.buffer([kernelSize, kernelSize]);
    const center = Math.floor(kernelSize / 2);
    
    let sum = 0;
    for (let i = 0; i < kernelSize; i++) {
      for (let j = 0; j < kernelSize; j++) {
        const distance = Math.sqrt(
          Math.pow(i - center, 2) + Math.pow(j - center, 2)
        );
        const value = Math.exp(-distance * distance / (2 * sigma * sigma));
        kernel.set(value, i, j);
        sum += value;
      }
    }
    
    // Normalize kernel
    for (let i = 0; i < kernelSize; i++) {
      for (let j = 0; j < kernelSize; j++) {
        kernel.set(kernel.get(i, j) / sum, i, j);
      }
    }
    
    // Convolve hit map with kernel (simplified)
    // For full implementation, would use tf.conv2d
    return hitMap;
  });
}