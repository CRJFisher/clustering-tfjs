import * as tf from '../tf-adapter';
import type { SOMTopology, SOMInitialization, SOMNeighborhood, DecayFunction } from './types';

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
    const [_nSamples, nFeatures] = X.shape;
    const _totalNeurons = gridHeight * gridWidth;
    
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
        const _centered = X.sub(mean);
        
        // Compute covariance matrix (simplified PCA)
        // Note: Full PCA implementation would use covariance matrix
        // const cov = tf.matMul(centered, centered, true, false).div(nSamples);
        
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
        // PCA-based initialization - place weights along principal components
        const [_nSamples, nFeatures] = X.shape;
        
        if (_nSamples < 2) {
          // Fall back to random initialization if not enough samples
          return initializeWeights(X, gridHeight, gridWidth, 'random', randomSeed);
        }
        
        // Center the data
        const mean = X.mean(0, true);
        const centered = X.sub(mean);
        
        // Compute covariance matrix
        const cov = tf.matMul(centered, centered, true, false).div(_nSamples - 1);
        
        // Get principal components
        const nComps = Math.min(2, nFeatures);
        const components = computePrincipalComponents(cov as tf.Tensor2D, nComps);
        
        // Project data onto principal components
        const projections = tf.matMul(centered, components, false, true);
        const pcMin = projections.min(0);
        const pcMax = projections.max(0);
        const pcRange = pcMax.sub(pcMin);
        
        // Create grid in PC space and transform back
        const weights: number[][][] = [];
        
        for (let i = 0; i < gridHeight; i++) {
          const rowWeights: number[][] = [];
          for (let j = 0; j < gridWidth; j++) {
            // Map grid position to PC space
            const alpha = gridHeight > 1 ? i / (gridHeight - 1) : 0.5;
            const beta = gridWidth > 1 ? j / (gridWidth - 1) : 0.5;
            
            // Create PC coordinates
            const pcCoords = tf.zeros([nComps]);
            const pcBuffer = pcCoords.bufferSync();
            pcBuffer.set(pcMin.dataSync()[0] + pcRange.dataSync()[0] * alpha, 0);
            if (nComps > 1) {
              pcBuffer.set(pcMin.dataSync()[1] + pcRange.dataSync()[1] * beta, 1);
            }
            
            // Transform back to original space
            const weight = tf.tidy(() => {
              const reconstructed = tf.matMul(
                pcBuffer.toTensor().expandDims(0),
                components
              );
              return reconstructed.squeeze().add(mean);
            });
            
            rowWeights.push(Array.from(weight.dataSync()));
            weight.dispose();
            pcCoords.dispose();
          }
          weights.push(rowWeights);
        }
        
        // Cleanup
        cov.dispose();
        components.dispose();
        projections.dispose();
        pcMin.dispose();
        pcMax.dispose();
        pcRange.dispose();
        centered.dispose();
        mean.dispose();
        
        return tf.tensor3d(weights);
      }
      
      default:
        throw new Error(`Unknown initialization strategy: ${initialization}`);
    }
  });
}

/**
 * Compute principal components using power iteration.
 * TensorFlow.js doesn't have eigendecomposition, so we use power iteration.
 * 
 * @param covMatrix Covariance matrix [nFeatures, nFeatures]
 * @param nComponents Number of components to extract
 * @returns Principal components [nComponents, nFeatures]
 */
function computePrincipalComponents(
  covMatrix: tf.Tensor2D,
  nComponents: number
): tf.Tensor2D {
  return tf.tidy(() => {
    const [n, _] = covMatrix.shape;
    const components: tf.Tensor1D[] = [];
    let deflatedMatrix = covMatrix.clone();
    
    for (let comp = 0; comp < Math.min(nComponents, n); comp++) {
      // Initialize random vector
      let v = tf.randomNormal([n, 1]);
      v = v.div(v.norm());
      
      // Power iteration
      for (let iter = 0; iter < 100; iter++) {
        const vNew = tf.matMul(deflatedMatrix, v);
        const norm = vNew.norm();
        
        if (norm.dataSync()[0] < 1e-10) break;
        
        v.dispose();
        v = vNew.div(norm);
      }
      
      // Store component
      components.push(v.squeeze());
      
      // Deflate matrix (remove component)
      const eigenvalue = tf.matMul(
        tf.matMul(v, deflatedMatrix, true, false),
        v
      ).squeeze();
      
      const outerProduct = tf.matMul(v, v, false, true);
      const deflation = outerProduct.mul(eigenvalue);
      
      const newDeflated = deflatedMatrix.sub(deflation) as tf.Tensor2D;
      deflatedMatrix.dispose();
      deflatedMatrix = newDeflated;
      
      // Cleanup
      eigenvalue.dispose();
      outerProduct.dispose();
      deflation.dispose();
      v.dispose();
    }
    
    deflatedMatrix.dispose();
    
    // Stack components into matrix
    return tf.stack(components) as tf.Tensor2D;
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
    const bmuIndexArray = bmuIndex.arraySync();
    const bmuIndexValue: number = Array.isArray(bmuIndexArray) 
      ? bmuIndexArray[0] as number
      : bmuIndexArray as number;
    
    // Convert flat index back to grid coordinates  
    const rowValue = Math.floor(bmuIndexValue / gridWidth);
    const colValue = bmuIndexValue % gridWidth;
    
    return tf.tensor1d([rowValue, colValue]);
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
    const [_nSamples, nFeatures] = samples.shape;
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
  // Get data outside tf.tidy to avoid disposal issues
  const [_nSamples, _nFeatures] = samples.shape;
  const [gridHeight, gridWidth, _] = weights.shape;
  const bmusArray = bmus.arraySync();
  
  return tf.tidy(() => {
    // Reshape weights for easier indexing
    const weightsFlat = weights.reshape([gridHeight * gridWidth, _nFeatures]);
    
    // Convert BMU coordinates to flat indices
    const bmuIndices: number[] = [];
    
    for (let i = 0; i < _nSamples; i++) {
      const row = bmusArray[i][0];
      const col = bmusArray[i][1];
      bmuIndices.push(row * gridWidth + col);
    }
    
    // Gather BMU weights using indices
    const bmuIndicesTensor = tf.tensor1d(bmuIndices, 'int32');
    const bmuWeights = tf.gather(weightsFlat, bmuIndicesTensor);
    
    // Compute distances
    const diff = samples.sub(bmuWeights);
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
    const [_nSamples, nFeatures] = samples.shape;
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
 * @returns Influence value with positive center and negative surround
 */
export function mexicanHatNeighborhood(
  distance: tf.Tensor,
  radius: number
): tf.Tensor {
  return tf.tidy(() => {
    // Ricker wavelet formula:
    // h(d, σ) = A * (1 - (d/σ)²) * exp(-(d/σ)²/2)
    // where A is normalization constant (we use 2 for stronger effect)
    
    const sigma = radius;
    const distNorm = distance.div(sigma);
    const distNormSquared = distNorm.square();
    
    // Core Ricker wavelet
    const term1 = tf.scalar(1).sub(distNormSquared);
    const term2 = distNormSquared.div(2).neg().exp();
    const wavelet = term1.mul(term2);
    
    // Apply amplitude scaling for stronger lateral inhibition
    const amplitude = 2.0;
    return wavelet.mul(amplitude);
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
    const [_nSamples] = bmus.shape;
    const totalNeurons = gridHeight * gridWidth;
    
    // Pre-compute grid distance matrix if not cached
    const gridDistances = createGridDistanceMatrix(gridHeight, gridWidth, topology);
    
    // Get BMU flat indices
    const bmusData = bmus.bufferSync();
    const bmuIndices: number[] = [];
    for (let i = 0; i < _nSamples; i++) {
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

/**
 * ----------------------------------------------------------------------------
 * Decay Strategies for Learning Rate and Radius
 * ----------------------------------------------------------------------------
 */

/**
 * Linear decay function.
 * Decreases linearly from initial to final value.
 * 
 * @param initial Initial value
 * @param final Final value (minimum)
 * @param epoch Current epoch
 * @param totalEpochs Total number of epochs
 * @returns Decayed value
 */
export function linearDecay(
  initial: number,
  final: number,
  epoch: number,
  totalEpochs: number
): number {
  if (totalEpochs <= 1) return initial;
  const progress = epoch / (totalEpochs - 1);
  return initial * (1 - progress) + final * progress;
}

/**
 * Exponential decay function.
 * Decreases exponentially with configurable decay rate.
 * 
 * @param initial Initial value
 * @param final Final value (minimum)
 * @param epoch Current epoch
 * @param totalEpochs Total number of epochs
 * @param decayRate Decay rate (default: 5)
 * @returns Decayed value
 */
export function exponentialDecay(
  initial: number,
  final: number,
  epoch: number,
  totalEpochs: number,
  decayRate: number = 5
): number {
  if (totalEpochs <= 1) return initial;
  const timeConstant = totalEpochs / decayRate;
  const decayFactor = Math.exp(-epoch / timeConstant);
  return final + (initial - final) * decayFactor;
}

/**
 * Inverse time decay function.
 * Decreases using inverse time formula.
 * 
 * @param initial Initial value
 * @param final Final value (minimum)
 * @param epoch Current epoch
 * @param totalEpochs Total number of epochs
 * @param decayRate Decay rate (default: 1)
 * @returns Decayed value
 */
export function inverseTimeDecay(
  initial: number,
  final: number,
  epoch: number,
  totalEpochs: number,
  decayRate: number = 1
): number {
  if (totalEpochs <= 1) return initial;
  const decayFactor = 1 / (1 + decayRate * epoch / totalEpochs);
  return final + (initial - final) * decayFactor;
}

/**
 * Create a decay scheduler for learning rate or radius.
 * 
 * @param initial Initial value or custom decay function
 * @param strategy Decay strategy name
 * @param totalEpochs Total number of epochs
 * @param final Final value (minimum)
 * @returns Decay function
 */
export function createDecayScheduler(
  initial: number | DecayFunction,
  strategy: 'linear' | 'exponential' | 'inverse' = 'exponential',
  totalEpochs: number,
  final?: number
): DecayFunction {
  // If already a function, return it
  if (typeof initial === 'function') {
    return initial;
  }
  
  // Set default final value
  const finalValue = final ?? initial * 0.01;
  
  // Return appropriate decay function
  switch (strategy) {
    case 'linear':
      return (epoch: number) => linearDecay(initial, finalValue, epoch, totalEpochs);
    case 'exponential':
      return (epoch: number) => exponentialDecay(initial, finalValue, epoch, totalEpochs);
    case 'inverse':
      return (epoch: number) => inverseTimeDecay(initial, finalValue, epoch, totalEpochs);
    default:
      throw new Error(`Unknown decay strategy: ${strategy}`);
  }
}

/**
 * Decay scheduler with TensorFlow.js tensors for GPU computation.
 * Used when decay needs to be computed on GPU for performance.
 * 
 * @param initial Initial value tensor
 * @param final Final value tensor
 * @param epoch Current epoch tensor
 * @param totalEpochs Total epochs tensor
 * @param strategy Decay strategy
 * @returns Decayed value tensor
 */
export function decayTensor(
  initial: tf.Tensor | number,
  final: tf.Tensor | number,
  epoch: tf.Tensor | number,
  totalEpochs: tf.Tensor | number,
  strategy: 'linear' | 'exponential' | 'inverse' = 'exponential'
): tf.Tensor {
  return tf.tidy(() => {
    // Convert to tensors if needed
    const initialT = typeof initial === 'number' ? tf.scalar(initial) : initial;
    const finalT = typeof final === 'number' ? tf.scalar(final) : final;
    const epochT = typeof epoch === 'number' ? tf.scalar(epoch) : epoch;
    const totalEpochsT = typeof totalEpochs === 'number' ? tf.scalar(totalEpochs) : totalEpochs;
    
    switch (strategy) {
      case 'linear': {
        // linear: initial * (1 - t) + final * t
        const progress = epochT.div(totalEpochsT.sub(1));
        const term1 = initialT.mul(tf.scalar(1).sub(progress));
        const term2 = finalT.mul(progress);
        return term1.add(term2);
      }
      
      case 'exponential': {
        // exponential: final + (initial - final) * exp(-epoch / timeConstant)
        const timeConstant = totalEpochsT.div(5); // decay rate = 5
        const decayFactor = epochT.div(timeConstant).neg().exp();
        return finalT.add(initialT.sub(finalT).mul(decayFactor));
      }
      
      case 'inverse': {
        // inverse: final + (initial - final) / (1 + epoch / totalEpochs)
        const decayFactor = tf.scalar(1).div(
          tf.scalar(1).add(epochT.div(totalEpochsT))
        );
        return finalT.add(initialT.sub(finalT).mul(decayFactor));
      }
      
      default:
        throw new Error(`Unknown decay strategy: ${strategy}`);
    }
  });
}

/**
 * Track decay schedule across epochs.
 * Maintains history of decayed values for visualization.
 */
export class DecayTracker {
  private history: number[] = [];
  private decayFn: DecayFunction;
  private currentEpoch: number = 0;
  
  constructor(
    initial: number | DecayFunction,
    strategy: 'linear' | 'exponential' | 'inverse' = 'exponential',
    totalEpochs: number,
    final?: number
  ) {
    this.decayFn = createDecayScheduler(initial, strategy, totalEpochs, final);
  }
  
  /**
   * Get current value and advance epoch.
   */
  next(totalEpochs: number): number {
    const value = this.decayFn(this.currentEpoch, totalEpochs);
    this.history.push(value);
    this.currentEpoch++;
    return value;
  }
  
  /**
   * Get current value without advancing.
   */
  current(totalEpochs: number): number {
    return this.decayFn(this.currentEpoch, totalEpochs);
  }
  
  /**
   * Reset tracker to initial state.
   */
  reset(): void {
    this.currentEpoch = 0;
    this.history = [];
  }
  
  /**
   * Get decay history.
   */
  getHistory(): number[] {
    return [...this.history];
  }
  
  /**
   * Get current epoch.
   */
  getEpoch(): number {
    return this.currentEpoch;
  }
}

/**
 * Calculate adaptive radius based on grid size.
 * 
 * @param gridHeight Grid height
 * @param gridWidth Grid width
 * @param epoch Current epoch
 * @param totalEpochs Total epochs
 * @returns Adaptive radius value
 */
export function adaptiveRadius(
  gridHeight: number,
  gridWidth: number,
  epoch: number,
  totalEpochs: number
): number {
  const initialRadius = Math.max(gridHeight, gridWidth) / 2;
  const finalRadius = 1;
  return exponentialDecay(initialRadius, finalRadius, epoch, totalEpochs);
}

/**
 * Calculate adaptive learning rate.
 * 
 * @param initial Initial learning rate
 * @param epoch Current epoch
 * @param totalEpochs Total epochs
 * @returns Adaptive learning rate
 */
export function adaptiveLearningRate(
  initial: number,
  epoch: number,
  totalEpochs: number
): number {
  const final = initial * 0.01;
  return exponentialDecay(initial, final, epoch, totalEpochs);
}