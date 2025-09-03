import { SOM } from '../../src/clustering/som';
import * as tf from '../../src/tf-adapter';
import {
  initializeWeights,
  findBMU,
  findBMUBatch,
  gaussianNeighborhood,
  bubbleNeighborhood,
  linearDecay,
  exponentialDecay,
  DecayTracker,
} from '../../src/clustering/som_utils';

describe('SOM', () => {
  beforeAll(() => {
    // Initialize TensorFlow.js backend
    tf.setBackend('cpu');
  });

  afterEach(() => {
    // Clean up tensors
    tf.disposeVariables();
  });

  describe('Basic functionality', () => {
    it('should create SOM with valid parameters', () => {
      const som = new SOM({
        gridWidth: 5,
        gridHeight: 5,
        nClusters: 25, // Compatibility param
        randomState: 42,
      });

      expect(som.params.gridWidth).toBe(5);
      expect(som.params.gridHeight).toBe(5);
      expect(som.params.topology).toBe('rectangular');
      expect(som.params.neighborhood).toBe('gaussian');
    });

    it('should throw error for invalid grid dimensions', () => {
      expect(() => new SOM({
        gridWidth: 0,
        gridHeight: 5,
        nClusters: 0,
      })).toThrow('gridWidth must be >= 1');

      expect(() => new SOM({
        gridWidth: 5,
        gridHeight: -1,
        nClusters: 5,
      })).toThrow('gridHeight must be >= 1');
    });

    it('should fit simple 2D data', async () => {
      const som = new SOM({
        gridWidth: 3,
        gridHeight: 3,
        nClusters: 9,
        numEpochs: 10,
        randomState: 42,
      });

      const X = tf.tensor2d([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
      ]);

      await som.fit(X);

      expect(som.weights_).toBeDefined();
      expect(som.labels_).toBeDefined();
      expect(som.labels_).toHaveLength(4);

      X.dispose();
    });

    it('should predict labels for new data', async () => {
      const som = new SOM({
        gridWidth: 3,
        gridHeight: 3,
        nClusters: 9,
        numEpochs: 10,
        randomState: 42,
      });

      const XTrain = tf.tensor2d([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
      ]);

      await som.fit(XTrain);

      const XTest = tf.tensor2d([
        [0.1, 0.1],
        [0.9, 0.9],
      ]);

      const labels = await som.predict(XTest);

      expect(labels).toBeDefined();
      expect(labels).toHaveLength(2);
      expect(Array.isArray(labels)).toBe(true);

      XTrain.dispose();
      XTest.dispose();
    });
  });

  describe('Grid initialization', () => {
    it('should initialize weights with random strategy', () => {
      const X = tf.tensor2d([
        [0, 0],
        [1, 1],
        [2, 2],
      ]);

      const weights = initializeWeights(X, 2, 2, 'random', 42);

      expect(weights.shape).toEqual([2, 2, 2]);
      
      const weightsArray = weights.arraySync();
      // Check weights are within data range
      expect(weightsArray[0][0][0]).toBeGreaterThanOrEqual(0);
      expect(weightsArray[0][0][0]).toBeLessThanOrEqual(2);

      X.dispose();
      weights.dispose();
    });

    it('should initialize weights with linear strategy', () => {
      const X = tf.tensor2d([
        [0, 0],
        [1, 1],
        [2, 2],
      ]);

      const weights = initializeWeights(X, 2, 2, 'linear');

      expect(weights.shape).toEqual([2, 2, 2]);

      X.dispose();
      weights.dispose();
    });

    it('should initialize weights with PCA strategy', () => {
      const X = tf.tensor2d([
        [0, 0],
        [1, 1],
        [2, 2],
      ]);

      const weights = initializeWeights(X, 2, 2, 'pca');

      expect(weights.shape).toEqual([2, 2, 2]);

      X.dispose();
      weights.dispose();
    });
  });

  describe('BMU calculation', () => {
    it('should find BMU for single sample', () => {
      const sample = tf.tensor1d([0.5, 0.5]);
      const weights = tf.tensor3d([
        [[0, 0], [1, 0]],
        [[0, 1], [1, 1]],
      ]);

      const bmu = findBMU(sample, weights);
      const bmuArray = bmu.arraySync();

      // BMU should be closest to [0.5, 0.5]
      expect(bmuArray).toHaveLength(2);
      expect(bmuArray[0]).toBeGreaterThanOrEqual(0);
      expect(bmuArray[0]).toBeLessThan(2);

      sample.dispose();
      weights.dispose();
      bmu.dispose();
    });

    it('should find BMUs for batch of samples', () => {
      const samples = tf.tensor2d([
        [0, 0],
        [1, 1],
      ]);
      const weights = tf.tensor3d([
        [[0, 0], [1, 0]],
        [[0, 1], [1, 1]],
      ]);

      const bmus = findBMUBatch(samples, weights);

      expect(bmus.shape).toEqual([2, 2]);

      samples.dispose();
      weights.dispose();
      bmus.dispose();
    });
  });

  describe('Neighborhood functions', () => {
    it('should calculate Gaussian neighborhood', () => {
      const distance = tf.tensor1d([0, 1, 2, 3]);
      const radius = 2;

      const influence = gaussianNeighborhood(distance, radius);
      const influenceArray = influence.arraySync() as number[];

      // Influence should decay with distance
      expect(influenceArray[0]).toBeCloseTo(1, 5);
      expect(influenceArray[1]).toBeLessThan(influenceArray[0]);
      expect(influenceArray[2]).toBeLessThan(influenceArray[1]);
      expect(influenceArray[3]).toBeLessThan(influenceArray[2]);

      distance.dispose();
      influence.dispose();
    });

    it('should calculate Bubble neighborhood', () => {
      const distance = tf.tensor1d([0, 1, 2, 3]);
      const radius = 2;

      const influence = bubbleNeighborhood(distance, radius);
      const influenceArray = influence.arraySync() as number[];

      // Bubble has hard cutoff
      expect(influenceArray[0]).toBe(1);
      expect(influenceArray[1]).toBe(1);
      expect(influenceArray[2]).toBe(1);
      expect(influenceArray[3]).toBe(0);

      distance.dispose();
      influence.dispose();
    });
  });

  describe('Decay strategies', () => {
    it('should apply linear decay', () => {
      const initial = 1.0;
      const final = 0.1;
      const totalEpochs = 10;

      const value0 = linearDecay(initial, final, 0, totalEpochs);
      const value5 = linearDecay(initial, final, 5, totalEpochs);
      const value9 = linearDecay(initial, final, 9, totalEpochs);

      expect(value0).toBeCloseTo(1.0, 5);
      expect(value5).toBeCloseTo(0.5, 5);  // Linear interpolation at midpoint
      expect(value9).toBeCloseTo(0.1, 5);
    });

    it('should apply exponential decay', () => {
      const initial = 1.0;
      const final = 0.1;
      const totalEpochs = 10;

      const value0 = exponentialDecay(initial, final, 0, totalEpochs);
      const valueMid = exponentialDecay(initial, final, 5, totalEpochs);
      const valueLate = exponentialDecay(initial, final, 9, totalEpochs);

      expect(value0).toBeCloseTo(1.0, 2);
      expect(valueMid).toBeLessThan(value0);
      expect(valueLate).toBeLessThan(valueMid);
      expect(valueLate).toBeGreaterThanOrEqual(final);
    });

    it('should track decay history', () => {
      const tracker = new DecayTracker(1.0, 'linear', 10, 0.1);

      const values = [];
      for (let i = 0; i < 5; i++) {
        values.push(tracker.next(10));
      }

      expect(tracker.getEpoch()).toBe(5);
      expect(tracker.getHistory()).toHaveLength(5);
      expect(values[0]).toBeGreaterThan(values[4]);
    });
  });

  describe('Online learning', () => {
    it('should support partial fit', async () => {
      const som = new SOM({
        gridWidth: 3,
        gridHeight: 3,
        nClusters: 9,
        numEpochs: 10,
        onlineMode: true,
        randomState: 42,
      });

      const batch1 = tf.tensor2d([
        [0, 0],
        [0, 1],
      ]);

      const batch2 = tf.tensor2d([
        [1, 0],
        [1, 1],
      ]);

      await som.partialFit(batch1);
      expect(som.getTotalSamplesLearned()).toBe(2);

      await som.partialFit(batch2);
      expect(som.getTotalSamplesLearned()).toBe(4);

      batch1.dispose();
      batch2.dispose();
    });

    it('should enable streaming mode', async () => {
      const som = new SOM({
        gridWidth: 3,
        gridHeight: 3,
        nClusters: 9,
        randomState: 42,
      });

      som.enableStreamingMode(16);

      expect(som.params.onlineMode).toBe(true);
      expect(som.params.miniBatchSize).toBe(16);

      const sample = tf.tensor2d([[0.5, 0.5]]);
      await som.processStream(sample);

      sample.dispose();
    });

    it('should provide streaming statistics', async () => {
      const som = new SOM({
        gridWidth: 3,
        gridHeight: 3,
        nClusters: 9,
        onlineMode: true,
        randomState: 42,
      });

      const batch = tf.tensor2d([
        [0, 0],
        [1, 1],
      ]);

      await som.partialFit(batch);

      const stats = som.getStreamingStats();
      expect(stats.totalSamples).toBe(2);
      expect(stats.virtualEpoch).toBeDefined();
      expect(stats.currentLearningRate).toBeGreaterThan(0);
      expect(stats.currentRadius).toBeGreaterThan(0);

      batch.dispose();
    });
  });

  describe('Model persistence', () => {
    it('should save and load state', async () => {
      const som = new SOM({
        gridWidth: 2,
        gridHeight: 2,
        nClusters: 4,
        numEpochs: 5,
        randomState: 42,
      });

      const X = tf.tensor2d([
        [0, 0],
        [1, 1],
      ]);

      await som.fit(X);

      // Save state
      const state = som.saveState();
      expect(state.weights).toBeDefined();
      expect(state.totalSamples).toBeGreaterThanOrEqual(0);

      // Create new SOM and load state
      const som2 = new SOM({
        gridWidth: 2,
        gridHeight: 2,
        nClusters: 4,
      });

      som2.loadState(state);
      expect(som2.weights_).toBeDefined();

      X.dispose();
    });

    it('should save and load from JSON', async () => {
      const som = new SOM({
        gridWidth: 2,
        gridHeight: 2,
        nClusters: 4,
        numEpochs: 5,
        randomState: 42,
      });

      const X = tf.tensor2d([
        [0, 0],
        [1, 1],
      ]);

      await som.fit(X);

      // Save to JSON
      const json = await som.saveToJSON();
      expect(json).toBeDefined();
      expect(typeof json).toBe('string');

      // Load from JSON
      const som2 = new SOM({
        gridWidth: 2,
        gridHeight: 2,
        nClusters: 4,
      });

      await som2.loadFromJSON(json);
      expect(som2.weights_).toBeDefined();

      X.dispose();
    });
  });

  describe('Quality metrics', () => {
    it('should calculate U-matrix', async () => {
      const som = new SOM({
        gridWidth: 3,
        gridHeight: 3,
        nClusters: 9,
        numEpochs: 10,
        randomState: 42,
      });

      const X = tf.tensor2d([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
      ]);

      await som.fit(X);

      const uMatrix = som.getUMatrix();
      expect(uMatrix.shape).toEqual([3, 3]);

      X.dispose();
      uMatrix.dispose();
    });

    it('should calculate quantization error', async () => {
      const som = new SOM({
        gridWidth: 3,
        gridHeight: 3,
        nClusters: 9,
        numEpochs: 10,
        randomState: 42,
      });

      const X = tf.tensor2d([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
      ]);

      await som.fit(X);

      const qError = som.quantizationError();
      expect(qError).toBeGreaterThan(0);
      expect(qError).toBeLessThan(10); // Reasonable range

      X.dispose();
    });
  });

  describe('Memory management', () => {
    it('should dispose tensors properly', async () => {
      const som = new SOM({
        gridWidth: 2,
        gridHeight: 2,
        nClusters: 4,
        numEpochs: 5,
        randomState: 42,
      });

      const X = tf.tensor2d([
        [0, 0],
        [1, 1],
      ]);

      const initialMemory = tf.memory().numTensors;
      
      await som.fit(X);
      
      som.dispose();
      X.dispose();

      const finalMemory = tf.memory().numTensors;
      
      // Should have cleaned up most tensors (allowing small tolerance)
      expect(finalMemory).toBeLessThanOrEqual(initialMemory + 4);
    });
  });
});