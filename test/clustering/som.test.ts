import { SOM } from '../../src/clustering/som';
import * as tf from '../../src/tf-adapter';
import {
  initializeWeights,
  findBMU,
  findBMUBatch,
  findSecondBMU,
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

      })).toThrow('gridWidth must be >= 1');

      expect(() => new SOM({
        gridWidth: 5,
        gridHeight: -1,

      })).toThrow('gridHeight must be >= 1');
    });

    it('should fit simple 2D data', async () => {
      const som = new SOM({
        gridWidth: 3,
        gridHeight: 3,

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

      });

      som2.loadState(state);
      expect(som2.weights_).toBeDefined();

      X.dispose();
    });

    it('should save and load from JSON', async () => {
      const som = new SOM({
        gridWidth: 2,
        gridHeight: 2,

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

  describe('Task-37 correctness fixes', () => {
    describe('AC#1: Weight update normalization', () => {
      it('should produce finite weights with large batch sizes', async () => {
        const som = new SOM({
          gridWidth: 2,
          gridHeight: 2,
          numEpochs: 1,
          randomState: 42,
          miniBatchSize: 100,
        });

        const X = tf.randomUniform([100, 3], -5, 5, 'float32', 42) as tf.Tensor2D;
        await som.fit(X);

        const weights = som.getWeights();
        for (const row of weights) {
          for (const neuron of row) {
            for (const val of neuron) {
              expect(isFinite(val)).toBe(true);
              expect(Math.abs(val)).toBeLessThan(100);
            }
          }
        }

        X.dispose();
        som.dispose();
      });
    });

    describe('AC#2: Rectangular 8-connectivity consistency', () => {
      it('should treat diagonal neighbors as neighbors in rectangular topology', () => {
        const som = new SOM({
          gridWidth: 3,
          gridHeight: 3,
        });

        // Diagonal neighbors should be neighbors (8-connectivity)
        expect(som['areNeighbors'](0, 0, 1, 1, 3, 3, 'rectangular')).toBe(true);
        expect(som['areNeighbors'](1, 1, 0, 0, 3, 3, 'rectangular')).toBe(true);
        expect(som['areNeighbors'](1, 1, 2, 2, 3, 3, 'rectangular')).toBe(true);

        // Non-adjacent should not be neighbors
        expect(som['areNeighbors'](0, 0, 2, 2, 3, 3, 'rectangular')).toBe(false);
        expect(som['areNeighbors'](0, 0, 0, 2, 3, 3, 'rectangular')).toBe(false);

        // Cardinal neighbors still work
        expect(som['areNeighbors'](0, 0, 0, 1, 3, 3, 'rectangular')).toBe(true);
        expect(som['areNeighbors'](0, 0, 1, 0, 3, 3, 'rectangular')).toBe(true);

        som.dispose();
      });
    });

    describe('AC#3: findSecondBMU iterative min-finding', () => {
      it('should find correct second BMU for known weights', () => {
        const sample = tf.tensor1d([0.9, 0.9]) as tf.Tensor1D;
        const weights = tf.tensor3d([
          [[0, 0], [1, 0]],
          [[0, 1], [1, 1]],
        ]) as tf.Tensor3D;

        // BMU should be [1,1] (weight [1,1]) since sample is [0.9, 0.9]
        const bmu = findBMU(sample, weights);
        const bmuArray = bmu.arraySync();
        expect(bmuArray[0]).toBe(1);
        expect(bmuArray[1]).toBe(1);

        // Second BMU should be one of the adjacent neurons
        const secondBmu = findSecondBMU(sample, weights, bmu);
        const secondArray = secondBmu.arraySync();

        // Second BMU should not be the same as BMU
        expect(secondArray[0] === bmuArray[0] && secondArray[1] === bmuArray[1]).toBe(false);

        sample.dispose();
        weights.dispose();
        bmu.dispose();
        secondBmu.dispose();
      });

      it('should not stack overflow on large grids', () => {
        // 50x50 = 2,500 neurons — would overflow with Math.min(...spread)
        const nFeatures = 2;
        const weights = tf.randomUniform([50, 50, nFeatures], 0, 1, 'float32', 42) as tf.Tensor3D;
        const sample = tf.randomUniform([nFeatures], 0, 1, 'float32', 42) as tf.Tensor1D;
        const bmu = findBMU(sample, weights);

        // Should not throw (the old spread approach would RangeError here)
        const secondBmu = findSecondBMU(sample, weights, bmu);
        const arr = secondBmu.arraySync();

        // Validate returned coordinates are within grid bounds
        expect(arr[0]).toBeGreaterThanOrEqual(0);
        expect(arr[0]).toBeLessThan(50);
        expect(arr[1]).toBeGreaterThanOrEqual(0);
        expect(arr[1]).toBeLessThan(50);

        secondBmu.dispose();
        sample.dispose();
        weights.dispose();
        bmu.dispose();
      });
    });

    describe('AC#4: Data shuffling between epochs', () => {
      it('should produce deterministic results with same randomState', async () => {
        const X = tf.tensor2d([
          [0, 0], [0, 1], [1, 0], [1, 1],
          [0.5, 0.5], [0.2, 0.8], [0.8, 0.2], [0.3, 0.3],
        ]);

        const som1 = new SOM({
          gridWidth: 3, gridHeight: 3,
          numEpochs: 10, randomState: 42,
        });
        const som2 = new SOM({
          gridWidth: 3, gridHeight: 3,
          numEpochs: 10, randomState: 42,
        });

        await som1.fit(X);
        await som2.fit(X);

        const w1 = som1.getWeights();
        const w2 = som2.getWeights();

        // Same seed should produce identical weights
        for (let i = 0; i < 3; i++) {
          for (let j = 0; j < 3; j++) {
            for (let k = 0; k < 2; k++) {
              expect(w1[i][j][k]).toBeCloseTo(w2[i][j][k], 5);
            }
          }
        }

        X.dispose();
        som1.dispose();
        som2.dispose();
      });
    });

    describe('AC#5: Linear initialization with PCA', () => {
      it('should span a 2D surface, not a 1D line', () => {
        // Data with clear 2-axis variance
        const X = tf.tensor2d([
          [3, 0, 0], [-3, 0, 0],
          [0, 2, 0], [0, -2, 0],
          [1, 1, 0], [-1, -1, 0],
        ]);

        const weights = initializeWeights(X, 3, 3, 'linear');
        const w = weights.arraySync();

        // Verify 2D surface: row direction and column direction should not be parallel
        const topLeft = w[0][0];
        const topRight = w[0][2];
        const bottomLeft = w[2][0];

        const rowDir = topRight.map((v, i) => v - topLeft[i]);
        const colDir = bottomLeft.map((v, i) => v - topLeft[i]);

        const magRow = Math.sqrt(rowDir.reduce((s, v) => s + v * v, 0));
        const magCol = Math.sqrt(colDir.reduce((s, v) => s + v * v, 0));

        // Both directions should have non-trivial magnitude
        expect(magRow).toBeGreaterThan(0.01);
        expect(magCol).toBeGreaterThan(0.01);

        // Directions should not be parallel (cosine < 0.5)
        const dot = rowDir.reduce((s, v, i) => s + v * colDir[i], 0);
        const cosAngle = Math.abs(dot / (magRow * magCol + 1e-10));
        expect(cosAngle).toBeLessThan(0.5);

        X.dispose();
        weights.dispose();
      });
    });

    describe('AC#6: getDensityMap Gaussian convolution', () => {
      it('should apply smoothing and preserve output shape', async () => {
        const som = new SOM({
          gridWidth: 3,
          gridHeight: 3,
          numEpochs: 10,
          randomState: 42,
        });

        const X = tf.tensor2d([
          [0, 0], [0, 1], [1, 0], [1, 1],
        ]);

        await som.fit(X);

        const { getDensityMap, getHitMap } = await import('../../src/utils/som_visualization');

        const hitMap = await getHitMap(som, X);
        const densityMap = await getDensityMap(som, X, 1.0);

        // Shape should be preserved
        expect(densityMap.shape).toEqual([3, 3]);

        const hitData = await hitMap.array();
        const densityData = await densityMap.array();

        // Density map should differ from raw hitMap (smoothing applied)
        let hasDifference = false;
        for (let i = 0; i < 3; i++) {
          for (let j = 0; j < 3; j++) {
            if (Math.abs(hitData[i][j] - densityData[i][j]) > 0.001) {
              hasDifference = true;
            }
          }
        }
        expect(hasDifference).toBe(true);

        // All density values should be non-negative
        for (const row of densityData) {
          for (const val of row) {
            expect(val).toBeGreaterThanOrEqual(0);
          }
        }

        hitMap.dispose();
        densityMap.dispose();
        X.dispose();
        som.dispose();
      });
    });
  });

  describe('Task-46: cluster(), validation, and API contracts', () => {

    describe('AC#1: SOM.cluster() method', () => {

      it('should throw if called before fit', async () => {
        const som = new SOM({ gridWidth: 3, gridHeight: 3 });
        await expect(som.cluster(2)).rejects.toThrow('SOM must be fitted before clustering');
        som.dispose();
      });

      it('should throw if nClusters < 1', async () => {
        const som = new SOM({
          gridWidth: 3, gridHeight: 3, numEpochs: 5, randomState: 42,
        });
        const X = tf.tensor2d([[0, 0], [1, 1], [2, 2], [3, 3]]);
        await som.fit(X);

        await expect(som.cluster(0)).rejects.toThrow('nClusters must be a positive integer');
        await expect(som.cluster(-1)).rejects.toThrow('nClusters must be a positive integer');

        X.dispose();
        som.dispose();
      });

      it('should throw if nClusters > total neurons', async () => {
        const som = new SOM({
          gridWidth: 2, gridHeight: 2, numEpochs: 5, randomState: 42,
        });
        const X = tf.tensor2d([[0, 0], [1, 1], [2, 2], [3, 3]]);
        await som.fit(X);

        await expect(som.cluster(5)).rejects.toThrow('exceeds total number of neurons (4)');

        X.dispose();
        som.dispose();
      });

      it('should return labels with correct length matching training data', async () => {
        const som = new SOM({
          gridWidth: 3, gridHeight: 3, numEpochs: 10, randomState: 42,
        });
        const X = tf.tensor2d([
          [0, 0], [0.1, 0.1], [1, 0], [1, 1],
          [0.5, 0.5], [0.2, 0.8], [0.8, 0.2], [0.3, 0.3],
        ]);
        await som.fit(X);

        const labels = await som.cluster(3);
        expect(Array.isArray(labels)).toBe(true);
        expect(labels.length).toBe(8);

        X.dispose();
        som.dispose();
      });

      it('should return labels with values in range [0, nClusters-1]', async () => {
        const som = new SOM({
          gridWidth: 3, gridHeight: 3, numEpochs: 10, randomState: 42,
        });
        const X = tf.tensor2d([
          [0, 0], [0.1, 0.1], [5, 5], [5.1, 5.1],
          [0, 5], [0.1, 5.1],
        ]);
        await som.fit(X);

        const labels = await som.cluster(3);
        for (const label of labels) {
          expect(label).toBeGreaterThanOrEqual(0);
          expect(label).toBeLessThan(3);
        }

        X.dispose();
        som.dispose();
      });

      it('should return exactly nClusters distinct label values for well-separated data', async () => {
        const som = new SOM({
          gridWidth: 5, gridHeight: 5, numEpochs: 50, randomState: 42,
        });
        const X = tf.tensor2d([
          // Blob A near origin
          [0, 0], [0.1, 0.1], [0.2, 0], [0, 0.2],
          // Blob B far away
          [10, 10], [10.1, 10.1], [10.2, 10], [10, 10.2],
          // Blob C third corner
          [0, 10], [0.1, 10.1], [0.2, 10], [0, 10.2],
        ]);
        await som.fit(X);

        const labels = await som.cluster(3);
        expect(new Set(labels).size).toBe(3);

        X.dispose();
        som.dispose();
      });

      it('should produce meaningful groupings for well-separated clusters', async () => {
        const som = new SOM({
          gridWidth: 5, gridHeight: 5, numEpochs: 50, randomState: 42,
        });
        const X = tf.tensor2d([
          // Blob A (indices 0-3)
          [0, 0], [0.1, 0.1], [0.2, 0], [0, 0.2],
          // Blob B (indices 4-7)
          [10, 10], [10.1, 10.1], [10.2, 10], [10, 10.2],
          // Blob C (indices 8-11)
          [0, 10], [0.1, 10.1], [0.2, 10], [0, 10.2],
        ]);
        await som.fit(X);

        const labels = await som.cluster(3);

        // Points within same blob should get same label
        expect(labels[0]).toBe(labels[1]);
        expect(labels[0]).toBe(labels[2]);
        expect(labels[0]).toBe(labels[3]);

        expect(labels[4]).toBe(labels[5]);
        expect(labels[4]).toBe(labels[6]);
        expect(labels[4]).toBe(labels[7]);

        expect(labels[8]).toBe(labels[9]);
        expect(labels[8]).toBe(labels[10]);
        expect(labels[8]).toBe(labels[11]);

        // Different blobs should get different labels
        expect(labels[0]).not.toBe(labels[4]);
        expect(labels[0]).not.toBe(labels[8]);
        expect(labels[4]).not.toBe(labels[8]);

        X.dispose();
        som.dispose();
      });

      it('should work with hexagonal topology', async () => {
        const som = new SOM({
          gridWidth: 3, gridHeight: 3, topology: 'hexagonal',
          numEpochs: 20, randomState: 42,
        });
        const X = tf.tensor2d([[0, 0], [0, 1], [1, 0], [1, 1]]);
        await som.fit(X);

        const labels = await som.cluster(2);
        expect(labels.length).toBe(4);
        expect(new Set(labels).size).toBe(2);

        X.dispose();
        som.dispose();
      });

      it('should accept custom linkage and metric options', async () => {
        const som = new SOM({
          gridWidth: 3, gridHeight: 3, numEpochs: 10, randomState: 42,
        });
        const X = tf.tensor2d([[0, 0], [0, 1], [1, 0], [1, 1]]);
        await som.fit(X);

        const labels = await som.cluster(2, { linkage: 'average', metric: 'euclidean' });
        expect(labels.length).toBe(4);
        expect(new Set(labels).size).toBe(2);

        X.dispose();
        som.dispose();
      });

      it('should return consistent results with same randomState', async () => {
        const X = tf.tensor2d([
          [0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5],
        ]);

        const som1 = new SOM({ gridWidth: 3, gridHeight: 3, numEpochs: 10, randomState: 42 });
        const som2 = new SOM({ gridWidth: 3, gridHeight: 3, numEpochs: 10, randomState: 42 });

        await som1.fit(X);
        await som2.fit(X);

        const labels1 = await som1.cluster(2);
        const labels2 = await som2.cluster(2);
        expect(labels1).toEqual(labels2);

        X.dispose();
        som1.dispose();
        som2.dispose();
      });
    });

    describe('AC#3: partialFit() dimension validation', () => {

      it('should accept first partialFit call with any feature dimension', async () => {
        const som = new SOM({
          gridWidth: 2, gridHeight: 2, onlineMode: true, randomState: 42,
        });
        const batch = tf.tensor2d([[1, 2, 3], [4, 5, 6]]);
        await som.partialFit(batch);
        expect(som.getTotalSamplesLearned()).toBe(2);

        batch.dispose();
        som.dispose();
      });

      it('should accept second partialFit call with same feature dimension', async () => {
        const som = new SOM({
          gridWidth: 2, gridHeight: 2, onlineMode: true, randomState: 42,
        });
        const batch1 = tf.tensor2d([[0, 0], [1, 1]]);
        const batch2 = tf.tensor2d([[2, 2], [3, 3]]);

        await som.partialFit(batch1);
        await som.partialFit(batch2);
        expect(som.getTotalSamplesLearned()).toBe(4);

        batch1.dispose();
        batch2.dispose();
        som.dispose();
      });

      it('should throw on second partialFit call with different feature dimension', async () => {
        const som = new SOM({
          gridWidth: 2, gridHeight: 2, onlineMode: true, randomState: 42,
        });
        const batch1 = tf.tensor2d([[0, 0], [1, 1]]);
        const batch2 = tf.tensor2d([[0, 0, 0], [1, 1, 1]]);

        await som.partialFit(batch1);
        await expect(som.partialFit(batch2)).rejects.toThrow('Feature dimension mismatch');

        batch1.dispose();
        batch2.dispose();
        som.dispose();
      });

      it('should include expected and actual dimensions in error message', async () => {
        const som = new SOM({
          gridWidth: 2, gridHeight: 2, onlineMode: true, randomState: 42,
        });
        const batch1 = tf.tensor2d([[0, 0], [1, 1]]);
        const batch2 = tf.tensor2d([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]);

        await som.partialFit(batch1);
        await expect(som.partialFit(batch2)).rejects.toThrow(
          /expected 2.*got 5/
        );

        batch1.dispose();
        batch2.dispose();
        som.dispose();
      });

      it('should validate dimension when fit() was called first', async () => {
        const som = new SOM({
          gridWidth: 2, gridHeight: 2, onlineMode: true,
          numEpochs: 5, randomState: 42,
        });
        const fitData = tf.tensor2d([[0, 0], [1, 1]]);
        await som.fit(fitData);

        const badBatch = tf.tensor2d([[0, 0, 0], [1, 1, 1]]);
        await expect(som.partialFit(badBatch)).rejects.toThrow('Feature dimension mismatch');

        fitData.dispose();
        badBatch.dispose();
        som.dispose();
      });
    });

    describe('AC#5: getWeights() contract', () => {

      it('should throw if called before fit', () => {
        const som = new SOM({ gridWidth: 2, gridHeight: 2 });
        expect(() => som.getWeights()).toThrow('SOM must be fitted first');
        som.dispose();
      });

      it('should return correct shape [gridHeight][gridWidth][nFeatures]', async () => {
        const som = new SOM({
          gridWidth: 4, gridHeight: 3, numEpochs: 5, randomState: 42,
        });
        const X = tf.tensor2d([[0, 0], [1, 1]]);
        await som.fit(X);

        const weights = som.getWeights();
        expect(weights.length).toBe(3);         // gridHeight
        expect(weights[0].length).toBe(4);       // gridWidth
        expect(weights[0][0].length).toBe(2);    // nFeatures

        X.dispose();
        som.dispose();
      });

      it('should return a plain number[][][] array, not a tensor', async () => {
        const som = new SOM({
          gridWidth: 2, gridHeight: 2, numEpochs: 5, randomState: 42,
        });
        const X = tf.tensor2d([[0, 0], [1, 1]]);
        await som.fit(X);

        const weights = som.getWeights();
        expect(Array.isArray(weights)).toBe(true);
        expect(Array.isArray(weights[0])).toBe(true);
        expect(Array.isArray(weights[0][0])).toBe(true);
        expect(typeof weights[0][0][0]).toBe('number');
        // Should NOT have tensor methods
        expect('dispose' in weights).toBe(false);
        expect('shape' in weights).toBe(false);

        X.dispose();
        som.dispose();
      });

      it('should return a snapshot not affected by further training', async () => {
        const som = new SOM({
          gridWidth: 2, gridHeight: 2, onlineMode: true,
          numEpochs: 5, randomState: 42,
        });
        const batch1 = tf.tensor2d([[0, 0], [1, 1]]);
        await som.partialFit(batch1);

        const weights1 = som.getWeights();

        const batch2 = tf.tensor2d([[10, 10], [20, 20]]);
        await som.partialFit(batch2);

        const weights2 = som.getWeights();

        // weights1 should be unchanged (snapshot)
        // weights2 should differ due to additional training
        let hasDifference = false;
        for (let i = 0; i < 2; i++) {
          for (let j = 0; j < 2; j++) {
            for (let k = 0; k < 2; k++) {
              if (Math.abs(weights1[i][j][k] - weights2[i][j][k]) > 1e-6) {
                hasDifference = true;
              }
            }
          }
        }
        expect(hasDifference).toBe(true);

        batch1.dispose();
        batch2.dispose();
        som.dispose();
      });

      it('should be safe to use after dispose()', async () => {
        const som = new SOM({
          gridWidth: 2, gridHeight: 2, numEpochs: 5, randomState: 42,
        });
        const X = tf.tensor2d([[0, 0], [1, 1]]);
        await som.fit(X);

        const weights = som.getWeights();
        som.dispose();

        // Plain array should still be valid after dispose
        expect(weights.length).toBe(2);
        expect(typeof weights[0][0][0]).toBe('number');

        X.dispose();
      });
    });

    describe('AC#6: dispose() behavior', () => {

      it('should release internal tensors on dispose', async () => {
        const som = new SOM({
          gridWidth: 2, gridHeight: 2, numEpochs: 5, randomState: 42,
        });
        const X = tf.tensor2d([[0, 0], [1, 1]]);
        await som.fit(X);

        const beforeDispose = tf.memory().numTensors;
        som.dispose();
        const afterDispose = tf.memory().numTensors;

        expect(afterDispose).toBeLessThan(beforeDispose);

        X.dispose();
      });

      it('should be safe to call dispose multiple times', async () => {
        const som = new SOM({
          gridWidth: 2, gridHeight: 2, numEpochs: 5, randomState: 42,
        });
        const X = tf.tensor2d([[0, 0], [1, 1]]);
        await som.fit(X);

        som.dispose();
        expect(() => som.dispose()).not.toThrow();

        X.dispose();
      });

      it('should cause getWeights to throw after dispose', async () => {
        const som = new SOM({
          gridWidth: 2, gridHeight: 2, numEpochs: 5, randomState: 42,
        });
        const X = tf.tensor2d([[0, 0], [1, 1]]);
        await som.fit(X);

        som.dispose();
        expect(() => som.getWeights()).toThrow('SOM must be fitted first');

        X.dispose();
      });
    });
  });
});