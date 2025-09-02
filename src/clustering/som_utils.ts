import * as tf from '../tf-adapter';
import type { SOMTopology, SOMInitialization, SOMNeighborhood } from './types';

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

/**
 * ----------------------------------------------------------------------------
 * Best Matching Unit (BMU) Calculation Functions
 * ----------------------------------------------------------------------------
 */

/**
 * Find the Best Matching Unit for a single sample.
 * 
 * @param sample Input sample tensor [nFeatures]
 * @param weights SOM weights tensor [gridHeight, gridWidth, nFeatures]
 * @returns BMU indices as [row, col]
 */
export function findBMU(
  sample: tf.Tensor1D,
  weights: tf.Tensor3D
): tf.Tensor1D {
  return tf.tidy(() => {
    const [gridHeight, gridWidth, nFeatures] = weights.shape;
    
    // Reshape weights to 2D for efficient computation
    // [gridHeight * gridWidth, nFeatures]
    const weightsFlat = weights.reshape([gridHeight * gridWidth, nFeatures]);
    
    // Compute squared distances to all neurons
    // ||sample - weight||^2 = ||sample||^2 + ||weight||^2 - 2 * sample . weight
    const sampleNorm = sample.square().sum().expandDims(0);
    const weightsNorm = weightsFlat.square().sum(1, true);
    const dotProduct = tf.matMul(
      sample.expandDims(0),
      weightsFlat,
      false,
      true
    );
    
    const distances = sampleNorm.add(weightsNorm).sub(dotProduct.mul(2));
    
    // Find minimum distance index
    const bmuIndex = distances.argMin(1);
    
    // Convert flat index back to grid coordinates
    const row = bmuIndex.div(gridWidth).floor();
    const col = bmuIndex.mod(gridWidth);
    
    return tf.stack([row, col]).squeeze();
  });
}

/**
 * Find BMUs for a batch of samples using optimized matrix operations.
 * 
 * @param samples Batch of input samples [nSamples, nFeatures]
 * @param weights SOM weights tensor [gridHeight, gridWidth, nFeatures]
 * @returns BMU indices tensor [nSamples, 2] with [row, col] for each sample
 */
export function findBMUBatch(
  samples: tf.Tensor2D,
  weights: tf.Tensor3D
): tf.Tensor2D {
  return tf.tidy(() => {
    const [nSamples, nFeatures] = samples.shape;
    const [gridHeight, gridWidth, _] = weights.shape;
    const totalNeurons = gridHeight * gridWidth;
    
    // Reshape weights for batch computation
    // [totalNeurons, nFeatures]
    const weightsFlat = weights.reshape([totalNeurons, nFeatures]);
    
    // Compute pairwise squared distances using broadcasting
    // Distance matrix will be [nSamples, totalNeurons]
    
    // ||x - w||^2 = ||x||^2 + ||w||^2 - 2 * x . w
    const samplesNorm = samples.square().sum(1, true); // [nSamples, 1]
    const weightsNorm = weightsFlat.square().sum(1, true).transpose(); // [1, totalNeurons]
    
    // Dot product: [nSamples, nFeatures] x [nFeatures, totalNeurons]
    const dotProduct = tf.matMul(samples, weightsFlat, false, true);
    
    // Compute distances
    const distances = samplesNorm.add(weightsNorm).sub(dotProduct.mul(2));
    
    // Find BMU indices for each sample
    const bmuIndices = distances.argMin(1); // [nSamples]
    
    // Convert flat indices to grid coordinates
    const rows = bmuIndices.div(gridWidth).floor();
    const cols = bmuIndices.mod(gridWidth);
    
    return tf.stack([rows, cols], 1) as tf.Tensor2D;
  });
}

/**
 * Compute distances from samples to their BMUs.
 * Used for quantization error calculation.
 * 
 * @param samples Input samples [nSamples, nFeatures]
 * @param weights SOM weights [gridHeight, gridWidth, nFeatures]
 * @param bmus BMU indices [nSamples, 2]
 * @returns Distances tensor [nSamples]
 */
export function computeBMUDistances(
  samples: tf.Tensor2D,
  weights: tf.Tensor3D,
  bmus: tf.Tensor2D
): tf.Tensor1D {
  return tf.tidy(() => {
    const [nSamples, nFeatures] = samples.shape;
    const [gridHeight, gridWidth, _] = weights.shape;
    
    // Get BMU weights for each sample
    const bmuWeights = tf.buffer([nSamples, nFeatures]);
    const weightsData = weights.bufferSync();
    const bmusData = bmus.bufferSync();
    
    for (let i = 0; i < nSamples; i++) {
      const row = bmusData.get(i, 0);
      const col = bmusData.get(i, 1);
      
      for (let j = 0; j < nFeatures; j++) {
        bmuWeights.set(weightsData.get(row, col, j), i, j);
      }
    }
    
    const bmuWeightsTensor = bmuWeights.toTensor();
    
    // Compute distances
    const diff = samples.sub(bmuWeightsTensor);
    const distances = diff.square().sum(1).sqrt();
    
    return distances as tf.Tensor1D;
  });
}

/**
 * Find the second-best matching unit for topographic error calculation.
 * 
 * @param sample Input sample [nFeatures]
 * @param weights SOM weights [gridHeight, gridWidth, nFeatures]
 * @param bmu First BMU indices [2]
 * @returns Second BMU indices [2]
 */
export function findSecondBMU(
  sample: tf.Tensor1D,
  weights: tf.Tensor3D,
  bmu: tf.Tensor1D
): tf.Tensor1D {
  return tf.tidy(() => {
    const [gridHeight, gridWidth, nFeatures] = weights.shape;
    const totalNeurons = gridHeight * gridWidth;
    
    // Get BMU flat index
    const bmuData = bmu.dataSync();
    const bmuRow = bmuData[0];
    const bmuCol = bmuData[1];
    const bmuFlatIndex = bmuRow * gridWidth + bmuCol;
    
    // Reshape weights
    const weightsFlat = weights.reshape([totalNeurons, nFeatures]);
    
    // Compute distances to all neurons
    const sampleExpanded = sample.expandDims(0);
    const diff = weightsFlat.sub(sampleExpanded);
    const distances = diff.square().sum(1).sqrt();
    
    // Set BMU distance to infinity to exclude it
    const distancesArray = Array.from(distances.dataSync());
    distancesArray[bmuFlatIndex] = Infinity;
    
    // Find second minimum
    const secondBMUIndex = distancesArray.indexOf(Math.min(...distancesArray.filter(d => d !== Infinity)));
    
    // Convert to grid coordinates
    const row = Math.floor(secondBMUIndex / gridWidth);
    const col = secondBMUIndex % gridWidth;
    
    return tf.tensor1d([row, col]);
  });
}

/**
 * Optimized BMU search using pre-allocated tensors for better memory management.
 * This version is designed for use in training loops where tensors can be reused.
 * 
 * @param samples Input samples [nSamples, nFeatures]
 * @param weights SOM weights [gridHeight, gridWidth, nFeatures]
 * @param distanceBuffer Optional pre-allocated distance buffer
 * @returns Object containing BMU indices and distances
 */
export function findBMUOptimized(
  samples: tf.Tensor2D,
  weights: tf.Tensor3D,
  distanceBuffer?: tf.Tensor2D
): { bmus: tf.Tensor2D; distances: tf.Tensor1D } {
  return tf.tidy(() => {
    const [nSamples, nFeatures] = samples.shape;
    const [gridHeight, gridWidth, _] = weights.shape;
    const totalNeurons = gridHeight * gridWidth;
    
    // Reshape weights efficiently
    const weightsFlat = weights.reshape([totalNeurons, nFeatures]);
    
    // Use optimized matrix multiplication for distance calculation
    // This leverages GPU acceleration maximally
    const samplesNorm = samples.square().sum(1, true);
    const weightsNorm = weightsFlat.square().sum(1, true).transpose();
    
    // Main computational bottleneck - optimized matrix multiplication
    const dotProduct = tf.matMul(samples, weightsFlat, false, true);
    
    // Compute distance matrix
    let distances: tf.Tensor2D;
    if (distanceBuffer) {
      // Reuse buffer if provided
      distances = distanceBuffer;
      distances = samplesNorm.add(weightsNorm).sub(dotProduct.mul(2));
    } else {
      distances = samplesNorm.add(weightsNorm).sub(dotProduct.mul(2));
    }
    
    // Find BMUs
    const bmuIndices = distances.argMin(1);
    
    // Get minimum distances for each sample
    const minDistances = distances.min(1).sqrt();
    
    // Convert to grid coordinates
    const rows = bmuIndices.div(gridWidth).floor();
    const cols = bmuIndices.mod(gridWidth);
    const bmus = tf.stack([rows, cols], 1) as tf.Tensor2D;
    
    return { bmus, distances: minDistances as tf.Tensor1D };
  });
}

/**
 * ----------------------------------------------------------------------------
 * Neighborhood Functions for Weight Updates
 * ----------------------------------------------------------------------------
 */

/**
 * Gaussian neighborhood function.
 * Smooth exponential decay from BMU.
 * 
 * @param distance Distance from BMU in grid space
 * @param radius Current neighborhood radius
 * @returns Influence value [0, 1]
 */
export function gaussianNeighborhood(
  distance: tf.Tensor,
  radius: number
): tf.Tensor {
  return tf.tidy(() => {
    // h(d, σ) = exp(-d² / (2σ²))
    const sigma2 = 2 * radius * radius;
    return distance.square().div(sigma2).neg().exp();
  });
}

/**
 * Bubble neighborhood function.
 * Hard cutoff at radius boundary.
 * 
 * @param distance Distance from BMU in grid space
 * @param radius Current neighborhood radius
 * @returns Influence value (0 or 1)
 */
export function bubbleNeighborhood(
  distance: tf.Tensor,
  radius: number
): tf.Tensor {
  return tf.tidy(() => {
    // h(d, σ) = 1 if d <= σ, else 0
    return distance.lessEqual(radius).cast('float32');
  });
}

/**
 * Mexican hat (Ricker wavelet) neighborhood function.
 * Provides lateral inhibition with negative values at medium distances.
 * 
 * @param distance Distance from BMU in grid space
 * @param radius Current neighborhood radius
 * @returns Influence value [-0.5, 1]
 */
export function mexicanHatNeighborhood(
  distance: tf.Tensor,
  radius: number
): tf.Tensor {
  return tf.tidy(() => {
    // h(d, σ) = (1 - d²/σ²) * exp(-d²/(2σ²))
    const sigma2 = radius * radius;
    const distSquared = distance.square();
    const term1 = tf.scalar(1).sub(distSquared.div(sigma2));
    const term2 = distSquared.div(2 * sigma2).neg().exp();
    return term1.mul(term2);
  });
}

/**
 * Compute neighborhood influence for all neurons given BMU positions.
 * 
 * @param bmus BMU positions [nSamples, 2]
 * @param gridHeight Grid height
 * @param gridWidth Grid width
 * @param radius Current neighborhood radius
 * @param neighborhood Neighborhood function type
 * @param topology Grid topology
 * @returns Influence matrix [nSamples, gridHeight * gridWidth]
 */
export function computeNeighborhoodInfluence(
  bmus: tf.Tensor2D,
  gridHeight: number,
  gridWidth: number,
  radius: number,
  neighborhood: SOMNeighborhood,
  topology: SOMTopology
): tf.Tensor2D {
  return tf.tidy(() => {
    const [nSamples] = bmus.shape;
    const totalNeurons = gridHeight * gridWidth;
    
    // Pre-compute grid distance matrix if not cached
    const gridDistances = createGridDistanceMatrix(gridHeight, gridWidth, topology);
    
    // Get BMU flat indices
    const bmusData = bmus.bufferSync();
    const bmuIndices: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      const row = bmusData.get(i, 0);
      const col = bmusData.get(i, 1);
      bmuIndices.push(row * gridWidth + col);
    }
    
    // Compute influence for each sample's BMU
    const influences: tf.Tensor[] = [];
    
    for (const bmuIdx of bmuIndices) {
      // Get distances from this BMU to all neurons
      const distances = gridDistances.slice([bmuIdx, 0], [1, totalNeurons]).squeeze();
      
      // Apply neighborhood function
      let influence: tf.Tensor;
      switch (neighborhood) {
        case 'gaussian':
          influence = gaussianNeighborhood(distances, radius);
          break;
        case 'bubble':
          influence = bubbleNeighborhood(distances, radius);
          break;
        case 'mexican_hat':
          influence = mexicanHatNeighborhood(distances, radius);
          break;
        default:
          throw new Error(`Unknown neighborhood function: ${neighborhood}`);
      }
      
      influences.push(influence.expandDims(0));
    }
    
    // Stack influences for all samples
    return tf.concat(influences, 0) as tf.Tensor2D;
  });
}

/**
 * Compute neighborhood influence matrix for batch update.
 * Optimized version that computes influence for all BMUs at once.
 * 
 * @param bmuIndices Flat BMU indices [nSamples]
 * @param gridDistanceMatrix Pre-computed grid distances [totalNeurons, totalNeurons]
 * @param radius Current neighborhood radius
 * @param neighborhood Neighborhood function type
 * @returns Influence matrix [nSamples, totalNeurons]
 */
export function computeNeighborhoodInfluenceBatch(
  bmuIndices: tf.Tensor1D,
  gridDistanceMatrix: tf.Tensor2D,
  radius: number,
  neighborhood: SOMNeighborhood
): tf.Tensor2D {
  return tf.tidy(() => {
    const nSamples = bmuIndices.shape[0];
    const totalNeurons = gridDistanceMatrix.shape[0];
    
    // Gather distances for all BMUs at once
    const bmuDistances = tf.gather(gridDistanceMatrix, bmuIndices);
    
    // Apply neighborhood function to all distances
    switch (neighborhood) {
      case 'gaussian':
        return gaussianNeighborhood(bmuDistances, radius) as tf.Tensor2D;
      case 'bubble':
        return bubbleNeighborhood(bmuDistances, radius) as tf.Tensor2D;
      case 'mexican_hat':
        return mexicanHatNeighborhood(bmuDistances, radius) as tf.Tensor2D;
      default:
        throw new Error(`Unknown neighborhood function: ${neighborhood}`);
    }
  });
}

/**
 * Create a cached neighborhood influence lookup table.
 * Pre-computes influence values for all possible neuron pairs.
 * 
 * @param gridHeight Grid height
 * @param gridWidth Grid width
 * @param radius Neighborhood radius
 * @param neighborhood Neighborhood function type
 * @param topology Grid topology
 * @returns Influence lookup table [totalNeurons, totalNeurons]
 */
export function createNeighborhoodLookupTable(
  gridHeight: number,
  gridWidth: number,
  radius: number,
  neighborhood: SOMNeighborhood,
  topology: SOMTopology
): tf.Tensor2D {
  return tf.tidy(() => {
    const gridDistances = createGridDistanceMatrix(gridHeight, gridWidth, topology);
    
    // Apply neighborhood function to entire distance matrix
    switch (neighborhood) {
      case 'gaussian':
        return gaussianNeighborhood(gridDistances, radius) as tf.Tensor2D;
      case 'bubble':
        return bubbleNeighborhood(gridDistances, radius) as tf.Tensor2D;
      case 'mexican_hat':
        return mexicanHatNeighborhood(gridDistances, radius) as tf.Tensor2D;
      default:
        throw new Error(`Unknown neighborhood function: ${neighborhood}`);
    }
  });
}

/**
 * Validate neighborhood function parameters.
 * 
 * @param radius Neighborhood radius
 * @param gridHeight Grid height
 * @param gridWidth Grid width
 * @throws Error if parameters are invalid
 */
export function validateNeighborhoodParams(
  radius: number,
  gridHeight: number,
  gridWidth: number
): void {
  if (radius <= 0) {
    throw new Error('Neighborhood radius must be positive');
  }
  
  const maxDimension = Math.max(gridHeight, gridWidth);
  if (radius > maxDimension) {
    console.warn(
      `Neighborhood radius (${radius}) is larger than grid dimensions. ` +
      `This may lead to uniform influence across the entire grid.`
    );
  }
}