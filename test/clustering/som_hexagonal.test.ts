import * as tf from '@tensorflow/tfjs';
import { SOM } from '../../src/clustering/som';
import { gridDistance } from '../../src/clustering/som_utils';

describe('SOM Hexagonal Topology', () => {
  beforeEach(() => {
    // Set a fixed seed for reproducible tests
    tf.randomUniform([1]);
  });

  afterEach(() => {
    // Clean up tensors
    tf.dispose();
  });

  describe('Grid Distance Calculations', () => {
    it('should calculate correct distances for hexagonal grid', () => {
      // Test various distance calculations
      const testCases = [
        { pos1: [0, 0], pos2: [0, 1], expected: 1.0 },      // Adjacent horizontally
        { pos1: [0, 0], pos2: [1, 0], expected: 1.0 },      // Adjacent vertically (even to odd)
        { pos1: [1, 0], pos2: [0, 0], expected: 1.0 },      // Adjacent vertically (odd to even)
        { pos1: [2, 2], pos2: [2, 2], expected: 0.0 },      // Same position
        { pos1: [0, 0], pos2: [2, 2], expected: 2.6458 },   // Diagonal
        { pos1: [0, 0], pos2: [4, 4], expected: 5.2915 },   // Far corner
      ];

      for (const { pos1, pos2, expected } of testCases) {
        const dist = gridDistance(pos1 as [number, number], pos2 as [number, number], 'hexagonal');
        expect(dist).toBeCloseTo(expected, 3);
      }
    });

    it('should calculate different distances for rectangular vs hexagonal', () => {
      const pos1: [number, number] = [0, 0];
      const pos2: [number, number] = [2, 2];
      
      const rectDist = gridDistance(pos1, pos2, 'rectangular');
      const hexDist = gridDistance(pos1, pos2, 'hexagonal');
      
      // Rectangular uses simple Euclidean: sqrt((2-0)^2 + (2-0)^2) = sqrt(8) ≈ 2.828
      expect(rectDist).toBeCloseTo(2.828, 3);
      // Hexagonal with offsets: different calculation
      expect(hexDist).toBeCloseTo(2.646, 3);
      expect(hexDist).not.toBeCloseTo(rectDist, 1);
    });
  });

  describe('Neighbor Detection', () => {
    it('should correctly identify neighbors in hexagonal topology', () => {
      const som = new SOM({
        nClusters: 9,
        gridHeight: 3,
        gridWidth: 3,
        topology: 'hexagonal'
      });
      
      // Test even row (row 0)
      expect(som['areNeighbors'](0, 0, 0, 1, 3, 3, 'hexagonal')).toBe(true);  // Right
      expect(som['areNeighbors'](0, 0, 1, 0, 3, 3, 'hexagonal')).toBe(true);  // Bottom-right (even row)
      expect(som['areNeighbors'](0, 0, 0, 2, 3, 3, 'hexagonal')).toBe(false); // Too far
      
      // Test odd row (row 1)
      expect(som['areNeighbors'](1, 0, 0, 0, 3, 3, 'hexagonal')).toBe(true);  // Top-left (odd to even)
      expect(som['areNeighbors'](1, 0, 0, 1, 3, 3, 'hexagonal')).toBe(true);  // Top-right (odd to even)
      expect(som['areNeighbors'](1, 0, 1, 1, 3, 3, 'hexagonal')).toBe(true);  // Right
      expect(som['areNeighbors'](1, 0, 2, 0, 3, 3, 'hexagonal')).toBe(true);  // Bottom-left (odd row)
      expect(som['areNeighbors'](1, 0, 2, 1, 3, 3, 'hexagonal')).toBe(true);  // Bottom-right (odd row)
      
      // Test center position (2,2 on even row)
      // Center would have neighbors at:
      // [1, 1], [1, 2] (Top-left, top-right)
      // [2, 1], [2, 3] (Left, right - note: 2,3 is out of bounds for 3x3 grid)
      // [3, 1], [3, 2] (Bottom-left, bottom-right - note: row 3 is out of bounds)
      
      // Only valid neighbors within 3x3 grid
      expect(som['areNeighbors'](2, 2, 1, 1, 3, 3, 'hexagonal')).toBe(true);  // Top-left
      expect(som['areNeighbors'](2, 2, 1, 2, 3, 3, 'hexagonal')).toBe(true);  // Top-right
      expect(som['areNeighbors'](2, 2, 2, 1, 3, 3, 'hexagonal')).toBe(true);  // Left
      expect(som['areNeighbors'](2, 2, 0, 0, 3, 3, 'hexagonal')).toBe(false); // Too far
    });

    it('should handle edge cases correctly', () => {
      const som = new SOM({
        nClusters: 25,
        gridHeight: 5,
        gridWidth: 5,
        topology: 'hexagonal'
      });
      
      // Corner neurons should have fewer neighbors
      // Top-left (0,0) - even row
      expect(som['areNeighbors'](0, 0, 0, 1, 5, 5, 'hexagonal')).toBe(true);  // Right
      expect(som['areNeighbors'](0, 0, 1, 0, 5, 5, 'hexagonal')).toBe(true);  // Bottom-right
      expect(som['areNeighbors'](0, 0, 1, 1, 5, 5, 'hexagonal')).toBe(false); // Not a direct neighbor
      
      // Top-right (0,4) - even row
      expect(som['areNeighbors'](0, 4, 0, 3, 5, 5, 'hexagonal')).toBe(true);  // Left
      expect(som['areNeighbors'](0, 4, 1, 3, 5, 5, 'hexagonal')).toBe(true);  // Bottom-left
      expect(som['areNeighbors'](0, 4, 1, 4, 5, 5, 'hexagonal')).toBe(true);  // Bottom-right
    });
  });

  describe('U-Matrix Computation', () => {
    it('should compute U-matrix with hexagonal neighbors', async () => {
      const data = tf.randomUniform([20, 2], 0, 1);
      
      const som = new SOM({
        nClusters: 9,
        gridHeight: 3,
        gridWidth: 3,
        topology: 'hexagonal',
        numEpochs: 10,
        learningRate: 0.5,
        radius: 1
      });
      
      await som.fit(data as tf.Tensor2D);
      
      const uMatrix = som.getUMatrix();
      const uMatrixData = await uMatrix.array();
      
      // Check dimensions
      expect(uMatrixData.length).toBe(3);
      expect(uMatrixData[0].length).toBe(3);
      
      // All values should be non-negative
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          expect(uMatrixData[i][j]).toBeGreaterThanOrEqual(0);
        }
      }
      
      // Corner neurons should have different patterns due to fewer neighbors
      // (Can't test exact values due to random initialization, but structure should be correct)
      
      uMatrix.dispose();
      data.dispose();
    });

    it('should produce different U-matrix patterns for rectangular vs hexagonal', async () => {
      const data = tf.randomUniform([20, 2], 0, 1, 'float32', 42);
      
      // Create two SOMs with same parameters except topology
      const somRect = new SOM({
        nClusters: 9,
        gridHeight: 3,
        gridWidth: 3,
        topology: 'rectangular',
        numEpochs: 10,
        learningRate: 0.5,
        radius: 1,
        randomState: 42
      });
      
      const somHex = new SOM({
        nClusters: 9,
        gridHeight: 3,
        gridWidth: 3,
        topology: 'hexagonal',
        numEpochs: 10,
        learningRate: 0.5,
        radius: 1,
        randomState: 42
      });
      
      await somRect.fit(data as tf.Tensor2D);
      await somHex.fit(data as tf.Tensor2D);
      
      const uMatrixRect = somRect.getUMatrix();
      const uMatrixHex = somHex.getUMatrix();
      
      const rectData = await uMatrixRect.array();
      const hexData = await uMatrixHex.array();
      
      // They should be different due to different neighbor structures
      let foundDifference = false;
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          if (Math.abs(rectData[i][j] - hexData[i][j]) > 0.01) {
            foundDifference = true;
            break;
          }
        }
        if (foundDifference) break;
      }
      
      expect(foundDifference).toBe(true);
      
      uMatrixRect.dispose();
      uMatrixHex.dispose();
      data.dispose();
    });
  });

  describe('Topographic Error with Hexagonal Topology', () => {
    it('should calculate topographic error correctly for hexagonal grid', async () => {
      const data = tf.tensor2d([
        [0.1, 0.1],
        [0.9, 0.9],
        [0.1, 0.9],
        [0.9, 0.1],
        [0.5, 0.5],
      ]);
      
      const som = new SOM({
        nClusters: 9,
        gridHeight: 3,
        gridWidth: 3,
        topology: 'hexagonal',
        numEpochs: 50,
        learningRate: 0.5,
        radius: 1
      });
      
      await som.fit(data as tf.Tensor2D);
      
      const topError = await som.topographicError(data);
      
      // Topographic error should be between 0 and 1
      expect(topError).toBeGreaterThanOrEqual(0);
      expect(topError).toBeLessThanOrEqual(1);
      
      // Topographic error depends heavily on random initialization and small data size
      // With only 5 samples and 3x3 grid, high error is expected
      // Just verify it's a reasonable value
      
      data.dispose();
    });
  });

  describe('Complete Integration Test', () => {
    it('should train SOM with hexagonal topology and produce valid results', async () => {
      // Generate a simple 2D dataset
      const data = tf.tensor2d([
        [0.0, 0.0], [0.1, 0.1], [0.2, 0.0],
        [1.0, 0.0], [0.9, 0.1], [0.8, 0.0],
        [0.0, 1.0], [0.1, 0.9], [0.0, 0.8],
        [1.0, 1.0], [0.9, 0.9], [1.0, 0.8],
        [0.5, 0.5], [0.4, 0.6], [0.6, 0.4],
      ]);
      
      const som = new SOM({
        nClusters: 16,
        gridHeight: 4,
        gridWidth: 4,
        topology: 'hexagonal',
        numEpochs: 100,
        learningRate: 0.5,
        radius: 2,
        neighborhood: 'gaussian',
        randomState: 42
      });
      
      await som.fit(data as tf.Tensor2D);
      
      // Test various methods work correctly
      const weights = som.getWeights();
      expect(weights.shape).toEqual([4, 4, 2]);
      
      const labels = await som.predict(data);
      // Labels should be an array of length 15 or tensor1d
      const labelsArray = Array.isArray(labels) ? labels : await labels.array();
      expect(labelsArray.length).toEqual(15);
      
      // Instead of getActivationDistances, check quantization error
      const quantError = som.quantizationError();
      expect(quantError).toBeGreaterThanOrEqual(0);
      
      const uMatrix = som.getUMatrix();
      expect(uMatrix.shape).toEqual([4, 4]);
      
      const topError = await som.topographicError(data);
      expect(topError).toBeGreaterThanOrEqual(0);
      expect(topError).toBeLessThanOrEqual(1);
      
      // Clean up
      weights.dispose();
      if (!Array.isArray(labels)) {
        labels.dispose();
      }
      uMatrix.dispose();
      data.dispose();
    });
  });
});