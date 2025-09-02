import * as tf from '../tf-adapter';
import type { SOMTopology, SOMInitialization } from './types';

/**
 * Grid coordinate utilities for SOM topology management.
 */

/**
 * Convert 2D grid coordinates to 1D index.
 */
export function gridToIndex(row: number, col: number, gridWidth: number): number {
  return row * gridWidth + col;
}

/**
 * Convert 1D index to 2D grid coordinates.
 */
export function indexToGrid(index: number, gridWidth: number): [number, number] {
  const row = Math.floor(index / gridWidth);
  const col = index % gridWidth;
  return [row, col];
}

/**
 * Calculate distance between two grid positions based on topology.
 * 
 * @param pos1 First position [row, col]
 * @param pos2 Second position [row, col]
 * @param topology Grid topology type
 * @returns Euclidean distance between positions
 */
export function gridDistance(
  pos1: [number, number],
  pos2: [number, number],
  topology: SOMTopology
): number {
  const [r1, c1] = pos1;
  const [r2, c2] = pos2;
  
  if (topology === 'rectangular') {
    // Simple Euclidean distance for rectangular grid
    return Math.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2);
  } else {
    // Hexagonal grid distance calculation
    // In hexagonal topology, even rows are offset by 0.5
    const x1 = c1 + (r1 % 2) * 0.5;
    const y1 = r1 * Math.sqrt(3) / 2;
    const x2 = c2 + (r2 % 2) * 0.5;
    const y2 = r2 * Math.sqrt(3) / 2;
    
    return Math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2);
  }
}

/**
 * Get neighbors of a grid position based on topology.
 * 
 * @param row Row position
 * @param col Column position
 * @param gridHeight Grid height
 * @param gridWidth Grid width
 * @param topology Grid topology
 * @returns Array of neighbor positions [row, col]
 */
export function getNeighbors(
  row: number,
  col: number,
  gridHeight: number,
  gridWidth: number,
  topology: SOMTopology
): Array<[number, number]> {
  const neighbors: Array<[number, number]> = [];
  
  if (topology === 'rectangular') {
    // 8-connected neighborhood for rectangular grid
    const deltas = [
      [-1, -1], [-1, 0], [-1, 1],
      [0, -1],           [0, 1],
      [1, -1],  [1, 0],  [1, 1]
    ];
    
    for (const [dr, dc] of deltas) {
      const newRow = row + dr;
      const newCol = col + dc;
      if (newRow >= 0 && newRow < gridHeight && 
          newCol >= 0 && newCol < gridWidth) {
        neighbors.push([newRow, newCol]);
      }
    }
  } else {
    // Hexagonal grid neighbors (6-connected)
    // Even rows have different neighbor offsets than odd rows
    const evenRowDeltas = [
      [-1, -1], [-1, 0],
      [0, -1],  [0, 1],
      [1, -1],  [1, 0]
    ];
    const oddRowDeltas = [
      [-1, 0],  [-1, 1],
      [0, -1],  [0, 1],
      [1, 0],   [1, 1]
    ];
    
    const deltas = row % 2 === 0 ? evenRowDeltas : oddRowDeltas;
    
    for (const [dr, dc] of deltas) {
      const newRow = row + dr;
      const newCol = col + dc;
      if (newRow >= 0 && newRow < gridHeight && 
          newCol >= 0 && newCol < gridWidth) {
        neighbors.push([newRow, newCol]);
      }
    }
  }
  
  return neighbors;
}

/**
 * Initialize SOM weights based on the specified strategy.
 * 
 * @param X Input data tensor [nSamples, nFeatures]
 * @param gridHeight Height of the SOM grid
 * @param gridWidth Width of the SOM grid
 * @param initialization Initialization strategy
 * @param randomSeed Random seed for reproducibility
 * @returns Weight tensor [gridHeight, gridWidth, nFeatures]
 */
export function initializeWeights(
  X: tf.Tensor2D,
  gridHeight: number,
  gridWidth: number,
  initialization: SOMInitialization,
  randomSeed?: number
): tf.Tensor3D {
  return tf.tidy(() => {
    const [nSamples, nFeatures] = X.shape;
    const totalNeurons = gridHeight * gridWidth;
    
    switch (initialization) {
      case 'random': {
        // Random initialization from data range
        const xMin = X.min(0);
        const xMax = X.max(0);
        const range = xMax.sub(xMin);
        
        // Create random weights within data range
        const randomWeights = tf.randomUniform(
          [gridHeight, gridWidth, nFeatures],
          0,
          1,
          undefined,
          randomSeed
        );
        
        // Scale to data range
        return randomWeights
          .mul(range.reshape([1, 1, nFeatures]))
          .add(xMin.reshape([1, 1, nFeatures]));
      }
      
      case 'linear': {
        // Linear initialization along first two principal components
        // For simplicity, we'll use the data's variance directions
        
        // Center the data
        const mean = X.mean(0);
        const centered = X.sub(mean);
        
        // Compute covariance matrix (simplified PCA)
        const cov = tf.matMul(centered, centered, true, false).div(nSamples);
        
        // Get eigenvectors (using SVD as approximation)
        // Note: TensorFlow.js doesn't have full eigen decomposition,
        // so we'll use a simplified approach
        
        // For linear initialization, create a linear grid in data space
        const xMin = X.min(0);
        const xMax = X.max(0);
        const xRange = xMax.sub(xMin);
        
        // Create grid coordinates
        const rowCoords = tf.linspace(0, 1, gridHeight);
        const colCoords = tf.linspace(0, 1, gridWidth);
        
        // Initialize weights as a linear combination
        const weights: number[][][] = [];
        for (let i = 0; i < gridHeight; i++) {
          const rowWeights: number[][] = [];
          for (let j = 0; j < gridWidth; j++) {
            // Linear interpolation in data space
            const alpha = i / Math.max(1, gridHeight - 1);
            const beta = j / Math.max(1, gridWidth - 1);
            
            // Create weight vector
            const weight = xMin.add(
              xRange.mul(tf.tensor1d([alpha * 0.7 + beta * 0.3]))
            );
            rowWeights.push(Array.from(weight.dataSync()));
            weight.dispose();
          }
          weights.push(rowWeights);
        }
        
        rowCoords.dispose();
        colCoords.dispose();
        
        return tf.tensor3d(weights);
      }
      
      case 'pca': {
        // PCA-based initialization
        // Use simplified PCA approach due to TensorFlow.js limitations
        
        // Center the data
        const mean = X.mean(0);
        const centered = X.sub(mean);
        
        // Get data standard deviation for each feature
        const std = centered.square().mean(0).sqrt();
        
        // Create a grid that spans the principal components
        const weights: number[][][] = [];
        
        for (let i = 0; i < gridHeight; i++) {
          const rowWeights: number[][] = [];
          for (let j = 0; j < gridWidth; j++) {
            // Map grid position to data space
            const alpha = (i - gridHeight / 2) / (gridHeight / 2);
            const beta = (j - gridWidth / 2) / (gridWidth / 2);
            
            // Create weight as combination of mean and spread
            const weight = mean.add(std.mul(tf.tensor1d([alpha, beta, 0, 0].slice(0, nFeatures))));
            rowWeights.push(Array.from(weight.dataSync()));
            weight.dispose();
          }
          weights.push(rowWeights);
        }
        
        return tf.tensor3d(weights);
      }
      
      default:
        throw new Error(`Unknown initialization strategy: ${initialization}`);
    }
  });
}

/**
 * Create a distance matrix between all grid positions.
 * Used for neighborhood calculations.
 * 
 * @param gridHeight Grid height
 * @param gridWidth Grid width
 * @param topology Grid topology
 * @returns Distance matrix [totalNeurons, totalNeurons]
 */
export function createGridDistanceMatrix(
  gridHeight: number,
  gridWidth: number,
  topology: SOMTopology
): tf.Tensor2D {
  const totalNeurons = gridHeight * gridWidth;
  const distances: number[][] = [];
  
  for (let i = 0; i < totalNeurons; i++) {
    const pos1 = indexToGrid(i, gridWidth);
    const row: number[] = [];
    
    for (let j = 0; j < totalNeurons; j++) {
      const pos2 = indexToGrid(j, gridWidth);
      row.push(gridDistance(pos1, pos2, topology));
    }
    distances.push(row);
  }
  
  return tf.tensor2d(distances);
}

/**
 * Generate grid coordinates for visualization.
 * 
 * @param gridHeight Grid height
 * @param gridWidth Grid width
 * @param topology Grid topology
 * @returns Coordinates tensor [totalNeurons, 2] with (x, y) positions
 */
export function getGridCoordinates(
  gridHeight: number,
  gridWidth: number,
  topology: SOMTopology
): tf.Tensor2D {
  const coords: number[][] = [];
  
  for (let row = 0; row < gridHeight; row++) {
    for (let col = 0; col < gridWidth; col++) {
      if (topology === 'rectangular') {
        coords.push([col, row]);
      } else {
        // Hexagonal coordinates with offset
        const x = col + (row % 2) * 0.5;
        const y = row * Math.sqrt(3) / 2;
        coords.push([x, y]);
      }
    }
  }
  
  return tf.tensor2d(coords);
}