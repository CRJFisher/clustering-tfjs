import { SOM } from '../../src/clustering/som';
import * as tf from '../../src/backend/adapter';
import {
  initialize_weights,
  find_bmu,
  find_bmu_batch,
  gaussian_neighborhood,
  bubble_neighborhood,
  linear_decay,
  exponential_decay,
  DecayTracker,
} from '../../src/clustering/som_neighborhood';

describe('SOM', () => {
  beforeAll(() => {
    // Initialize TensorFlow.js backend
    tf.set_backend('cpu');
  });

  afterEach(() => {
    // Clean up tensors
    tf.dispose_variables();
  });

  describe('Basic functionality', () => {
    it('should create SOM with valid parameters', () => {
      const som = new SOM({
        grid_width: 5,
        grid_height: 5,

        random_state: 42,
      });

      expect(som.params.grid_width).toBe(5);
      expect(som.params.grid_height).toBe(5);
      expect(som.params.topology).toBe('rectangular');
      expect(som.params.neighborhood).toBe('gaussian');
    });

    it('should throw error for invalid grid dimensions', () => {
      expect(() => new SOM({
        grid_width: 0,
        grid_height: 5,

      })).toThrow('gridWidth must be >= 1');

      expect(() => new SOM({
        grid_width: 5,
        grid_height: -1,

      })).toThrow('gridHeight must be >= 1');
    });

    it('should fit simple 2D data', async () => {
      const som = new SOM({
        grid_width: 3,
        grid_height: 3,

        num_epochs: 10,
        random_state: 42,
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
        grid_width: 3,
        grid_height: 3,

        num_epochs: 10,
        random_state: 42,
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

      const weights = initialize_weights(X, 2, 2, 'random', 42);

      expect(weights.shape).toEqual([2, 2, 2]);
      
      const weights_array = weights.arraySync();
      // Check weights are within data range
      expect(weights_array[0][0][0]).toBeGreaterThanOrEqual(0);
      expect(weights_array[0][0][0]).toBeLessThanOrEqual(2);

      X.dispose();
      weights.dispose();
    });

    it('should initialize weights with linear strategy', () => {
      const X = tf.tensor2d([
        [0, 0],
        [1, 1],
        [2, 2],
      ]);

      const weights = initialize_weights(X, 2, 2, 'linear');

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

      const weights = initialize_weights(X, 2, 2, 'pca');

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

      const bmu = find_bmu(sample, weights);
      const bmu_array = bmu.arraySync();

      // BMU should be closest to [0.5, 0.5]
      expect(bmu_array).toHaveLength(2);
      expect(bmu_array[0]).toBeGreaterThanOrEqual(0);
      expect(bmu_array[0]).toBeLessThan(2);

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

      const bmus = find_bmu_batch(samples, weights);

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

      const influence = gaussian_neighborhood(distance, radius);
      const influence_array = influence.arraySync() as number[];

      // Influence should decay with distance
      expect(influence_array[0]).toBeCloseTo(1, 5);
      expect(influence_array[1]).toBeLessThan(influence_array[0]);
      expect(influence_array[2]).toBeLessThan(influence_array[1]);
      expect(influence_array[3]).toBeLessThan(influence_array[2]);

      distance.dispose();
      influence.dispose();
    });

    it('should calculate Bubble neighborhood', () => {
      const distance = tf.tensor1d([0, 1, 2, 3]);
      const radius = 2;

      const influence = bubble_neighborhood(distance, radius);
      const influence_array = influence.arraySync() as number[];

      // Bubble has hard cutoff
      expect(influence_array[0]).toBe(1);
      expect(influence_array[1]).toBe(1);
      expect(influence_array[2]).toBe(1);
      expect(influence_array[3]).toBe(0);

      distance.dispose();
      influence.dispose();
    });
  });

  describe('Decay strategies', () => {
    it('should apply linear decay', () => {
      const initial = 1.0;
      const final = 0.1;
      const total_epochs = 10;

      const value0 = linear_decay(initial, final, 0, total_epochs);
      const value5 = linear_decay(initial, final, 5, total_epochs);
      const value9 = linear_decay(initial, final, 9, total_epochs);

      expect(value0).toBeCloseTo(1.0, 5);
      expect(value5).toBeCloseTo(0.5, 5);  // Linear interpolation at midpoint
      expect(value9).toBeCloseTo(0.1, 5);
    });

    it('should apply exponential decay', () => {
      const initial = 1.0;
      const final = 0.1;
      const total_epochs = 10;

      const value0 = exponential_decay(initial, final, 0, total_epochs);
      const value_mid = exponential_decay(initial, final, 5, total_epochs);
      const value_late = exponential_decay(initial, final, 9, total_epochs);

      expect(value0).toBeCloseTo(1.0, 2);
      expect(value_mid).toBeLessThan(value0);
      expect(value_late).toBeLessThan(value_mid);
      expect(value_late).toBeGreaterThanOrEqual(final);
    });

    it('should track decay history', () => {
      const tracker = new DecayTracker(1.0, 'linear', 10, 0.1);

      const values = [];
      for (let i = 0; i < 5; i++) {
        values.push(tracker.next(10));
      }

      expect(tracker.get_epoch()).toBe(5);
      expect(tracker.get_history()).toHaveLength(5);
      expect(values[0]).toBeGreaterThan(values[4]);
    });
  });

  describe('Online learning', () => {
    it('should support partial fit', async () => {
      const som = new SOM({
        grid_width: 3,
        grid_height: 3,

        num_epochs: 10,
        online_mode: true,
        random_state: 42,
      });

      const batch1 = tf.tensor2d([
        [0, 0],
        [0, 1],
      ]);

      const batch2 = tf.tensor2d([
        [1, 0],
        [1, 1],
      ]);

      await som.partial_fit(batch1);
      expect(som.get_total_samples_learned()).toBe(2);

      await som.partial_fit(batch2);
      expect(som.get_total_samples_learned()).toBe(4);

      batch1.dispose();
      batch2.dispose();
    });

    it('should enable streaming mode', async () => {
      const som = new SOM({
        grid_width: 3,
        grid_height: 3,

        random_state: 42,
      });

      som.enable_streaming_mode(16);

      expect(som.params.online_mode).toBe(true);
      expect(som.params.mini_batch_size).toBe(16);

      const sample = tf.tensor2d([[0.5, 0.5]]);
      await som.process_stream(sample);

      sample.dispose();
    });

    it('should provide streaming statistics', async () => {
      const som = new SOM({
        grid_width: 3,
        grid_height: 3,

        online_mode: true,
        random_state: 42,
      });

      const batch = tf.tensor2d([
        [0, 0],
        [1, 1],
      ]);

      await som.partial_fit(batch);

      const stats = som.get_streaming_stats();
      expect(stats.total_samples).toBe(2);
      expect(stats.virtual_epoch).toBeDefined();
      expect(stats.current_learning_rate).toBeGreaterThan(0);
      expect(stats.current_radius).toBeGreaterThan(0);

      batch.dispose();
    });
  });

  describe('Model persistence', () => {
    it('should save and load state', async () => {
      const som = new SOM({
        grid_width: 2,
        grid_height: 2,

        num_epochs: 5,
        random_state: 42,
      });

      const X = tf.tensor2d([
        [0, 0],
        [1, 1],
      ]);

      await som.fit(X);

      // Save state
      const state = som.save_state();
      expect(state.weights).toBeDefined();
      expect(state.total_samples).toBeGreaterThanOrEqual(0);

      // Create new SOM and load state
      const som2 = new SOM({
        grid_width: 2,
        grid_height: 2,

      });

      som2.load_state(state);
      expect(som2.weights_).toBeDefined();

      X.dispose();
    });

    it('should save and load from JSON', async () => {
      const som = new SOM({
        grid_width: 2,
        grid_height: 2,

        num_epochs: 5,
        random_state: 42,
      });

      const X = tf.tensor2d([
        [0, 0],
        [1, 1],
      ]);

      await som.fit(X);

      // Save to JSON
      const json = await som.save_to_json();
      expect(json).toBeDefined();
      expect(typeof json).toBe('string');

      // Load from JSON
      const som2 = new SOM({
        grid_width: 2,
        grid_height: 2,

      });

      await som2.load_from_json(json);
      expect(som2.weights_).toBeDefined();

      X.dispose();
    });
  });

  describe('Quality metrics', () => {
    it('should calculate U-matrix', async () => {
      const som = new SOM({
        grid_width: 3,
        grid_height: 3,

        num_epochs: 10,
        random_state: 42,
      });

      const X = tf.tensor2d([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
      ]);

      await som.fit(X);

      const u_matrix = som.get_u_matrix();
      expect(u_matrix.shape).toEqual([3, 3]);

      X.dispose();
      u_matrix.dispose();
    });

    it('should calculate quantization error', async () => {
      const som = new SOM({
        grid_width: 3,
        grid_height: 3,

        num_epochs: 10,
        random_state: 42,
      });

      const X = tf.tensor2d([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
      ]);

      await som.fit(X);

      const q_error = som.quantization_error();
      expect(q_error).toBeGreaterThan(0);
      expect(q_error).toBeLessThan(10); // Reasonable range

      X.dispose();
    });
  });

  describe('Memory management', () => {
    it('should dispose tensors properly', async () => {
      const som = new SOM({
        grid_width: 2,
        grid_height: 2,

        num_epochs: 5,
        random_state: 42,
      });

      const X = tf.tensor2d([
        [0, 0],
        [1, 1],
      ]);

      const initial_memory = tf.memory().numTensors;
      
      await som.fit(X);
      
      som.dispose();
      X.dispose();

      const final_memory = tf.memory().numTensors;
      
      // Should have cleaned up most tensors (allowing small tolerance)
      expect(final_memory).toBeLessThanOrEqual(initial_memory + 4);
    });
  });

  describe('Task-37 correctness fixes', () => {
    describe('AC#1: Weight update normalization', () => {
      it('should produce finite weights with large batch sizes', async () => {
        const som = new SOM({
          grid_width: 2,
          grid_height: 2,
          num_epochs: 1,
          random_state: 42,
          mini_batch_size: 100,
        });

        const X = tf.random_uniform([100, 3], -5, 5, 'float32', 42) as tf.Tensor2D;
        await som.fit(X);

        const weights = som.get_weights();
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
          grid_width: 3,
          grid_height: 3,
        });

        // Diagonal neighbors should be neighbors (8-connectivity)
        expect(som['are_neighbors'](0, 0, 1, 1, 3, 3, 'rectangular')).toBe(true);
        expect(som['are_neighbors'](1, 1, 0, 0, 3, 3, 'rectangular')).toBe(true);
        expect(som['are_neighbors'](1, 1, 2, 2, 3, 3, 'rectangular')).toBe(true);

        // Non-adjacent should not be neighbors
        expect(som['are_neighbors'](0, 0, 2, 2, 3, 3, 'rectangular')).toBe(false);
        expect(som['are_neighbors'](0, 0, 0, 2, 3, 3, 'rectangular')).toBe(false);

        // Cardinal neighbors still work
        expect(som['are_neighbors'](0, 0, 0, 1, 3, 3, 'rectangular')).toBe(true);
        expect(som['are_neighbors'](0, 0, 1, 0, 3, 3, 'rectangular')).toBe(true);

        som.dispose();
      });
    });

    describe('AC#4: Data shuffling between epochs', () => {
      it('should produce deterministic results with same randomState', async () => {
        const X = tf.tensor2d([
          [0, 0], [0, 1], [1, 0], [1, 1],
          [0.5, 0.5], [0.2, 0.8], [0.8, 0.2], [0.3, 0.3],
        ]);

        const som1 = new SOM({
          grid_width: 3, grid_height: 3,
          num_epochs: 10, random_state: 42,
        });
        const som2 = new SOM({
          grid_width: 3, grid_height: 3,
          num_epochs: 10, random_state: 42,
        });

        await som1.fit(X);
        await som2.fit(X);

        const w1 = som1.get_weights();
        const w2 = som2.get_weights();

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

        const weights = initialize_weights(X, 3, 3, 'linear');
        const w = weights.arraySync();

        // Verify 2D surface: row direction and column direction should not be parallel
        const top_left = w[0][0];
        const top_right = w[0][2];
        const bottom_left = w[2][0];

        const row_dir = top_right.map((v, i) => v - top_left[i]);
        const col_dir = bottom_left.map((v, i) => v - top_left[i]);

        const mag_row = Math.sqrt(row_dir.reduce((s, v) => s + v * v, 0));
        const mag_col = Math.sqrt(col_dir.reduce((s, v) => s + v * v, 0));

        // Both directions should have non-trivial magnitude
        expect(mag_row).toBeGreaterThan(0.01);
        expect(mag_col).toBeGreaterThan(0.01);

        // Directions should not be parallel (cosine < 0.5)
        const dot = row_dir.reduce((s, v, i) => s + v * col_dir[i], 0);
        const cos_angle = Math.abs(dot / (mag_row * mag_col + 1e-10));
        expect(cos_angle).toBeLessThan(0.5);

        X.dispose();
        weights.dispose();
      });
    });

    describe('AC#6: getDensityMap Gaussian convolution', () => {
      it('should apply smoothing and preserve output shape', async () => {
        const som = new SOM({
          grid_width: 3,
          grid_height: 3,
          num_epochs: 10,
          random_state: 42,
        });

        const X = tf.tensor2d([
          [0, 0], [0, 1], [1, 0], [1, 1],
        ]);

        await som.fit(X);

        const { get_density_map, get_hit_map } = await import('../../src/visualization/som_visualization');

        const hit_map = await get_hit_map(som, X);
        const density_map = await get_density_map(som, X, 1.0);

        // Shape should be preserved
        expect(density_map.shape).toEqual([3, 3]);

        const hit_data = await hit_map.array();
        const density_data = await density_map.array();

        // Density map should differ from raw hitMap (smoothing applied)
        let has_difference = false;
        for (let i = 0; i < 3; i++) {
          for (let j = 0; j < 3; j++) {
            if (Math.abs(hit_data[i][j] - density_data[i][j]) > 0.001) {
              has_difference = true;
            }
          }
        }
        expect(has_difference).toBe(true);

        // All density values should be non-negative
        for (const row of density_data) {
          for (const val of row) {
            expect(val).toBeGreaterThanOrEqual(0);
          }
        }

        hit_map.dispose();
        density_map.dispose();
        X.dispose();
        som.dispose();
      });
    });
  });

  describe('Task-46: cluster(), validation, and API contracts', () => {

    describe('AC#1: SOM.cluster() method', () => {

      it('should throw if called before fit', async () => {
        const som = new SOM({ grid_width: 3, grid_height: 3 });
        await expect(som.cluster(2)).rejects.toThrow('SOM must be fitted before clustering');
        som.dispose();
      });

      it('should throw if nClusters is not an integer', async () => {
        const som = new SOM({
          grid_width: 3, grid_height: 3, num_epochs: 5, random_state: 42,
        });
        const X = tf.tensor2d([[0, 0], [1, 1], [2, 2], [3, 3]]);
        await som.fit(X);

        await expect(som.cluster(2.5)).rejects.toThrow('nClusters must be a positive integer');

        X.dispose();
        som.dispose();
      });

      it('should throw if nClusters < 1', async () => {
        const som = new SOM({
          grid_width: 3, grid_height: 3, num_epochs: 5, random_state: 42,
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
          grid_width: 2, grid_height: 2, num_epochs: 5, random_state: 42,
        });
        const X = tf.tensor2d([[0, 0], [1, 1], [2, 2], [3, 3]]);
        await som.fit(X);

        await expect(som.cluster(5)).rejects.toThrow('exceeds total number of neurons (4)');

        X.dispose();
        som.dispose();
      });

      it('should return labels with correct length matching training data', async () => {
        const som = new SOM({
          grid_width: 3, grid_height: 3, num_epochs: 10, random_state: 42,
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
          grid_width: 3, grid_height: 3, num_epochs: 10, random_state: 42,
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
          grid_width: 5, grid_height: 5, num_epochs: 50, random_state: 42,
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
          grid_width: 5, grid_height: 5, num_epochs: 50, random_state: 42,
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
          grid_width: 3, grid_height: 3, topology: 'hexagonal',
          num_epochs: 20, random_state: 42,
        });
        const X = tf.tensor2d([[0, 0], [0, 1], [1, 0], [1, 1]]);
        await som.fit(X);

        const labels = await som.cluster(2);
        expect(labels.length).toBe(4);
        expect(new Set(labels).size).toBe(2);

        X.dispose();
        som.dispose();
      });

      it('should work with nClusters === 1 (all points in one cluster)', async () => {
        const som = new SOM({
          grid_width: 3, grid_height: 3, num_epochs: 10, random_state: 42,
        });
        const X = tf.tensor2d([[0, 0], [0, 1], [1, 0], [1, 1]]);
        await som.fit(X);

        const labels = await som.cluster(1);
        expect(labels.length).toBe(4);
        expect(new Set(labels).size).toBe(1);
        expect(labels[0]).toBe(0);

        X.dispose();
        som.dispose();
      });

      it('should work with nClusters === totalNeurons', async () => {
        const som = new SOM({
          grid_width: 2, grid_height: 2, num_epochs: 10, random_state: 42,
        });
        const X = tf.tensor2d([[0, 0], [0, 1], [1, 0], [1, 1]]);
        await som.fit(X);

        const labels = await som.cluster(4); // 2x2 = 4 neurons
        expect(labels.length).toBe(4);
        // Each label should be in range [0, 3]
        for (const label of labels) {
          expect(label).toBeGreaterThanOrEqual(0);
          expect(label).toBeLessThan(4);
        }

        X.dispose();
        som.dispose();
      });

      it('should accept custom linkage and metric options', async () => {
        const som = new SOM({
          grid_width: 3, grid_height: 3, num_epochs: 10, random_state: 42,
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

        const som1 = new SOM({ grid_width: 3, grid_height: 3, num_epochs: 10, random_state: 42 });
        const som2 = new SOM({ grid_width: 3, grid_height: 3, num_epochs: 10, random_state: 42 });

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
          grid_width: 2, grid_height: 2, online_mode: true, random_state: 42,
        });
        const batch = tf.tensor2d([[1, 2, 3], [4, 5, 6]]);
        await som.partial_fit(batch);
        expect(som.get_total_samples_learned()).toBe(2);

        batch.dispose();
        som.dispose();
      });

      it('should accept second partialFit call with same feature dimension', async () => {
        const som = new SOM({
          grid_width: 2, grid_height: 2, online_mode: true, random_state: 42,
        });
        const batch1 = tf.tensor2d([[0, 0], [1, 1]]);
        const batch2 = tf.tensor2d([[2, 2], [3, 3]]);

        await som.partial_fit(batch1);
        await som.partial_fit(batch2);
        expect(som.get_total_samples_learned()).toBe(4);

        batch1.dispose();
        batch2.dispose();
        som.dispose();
      });

      it('should throw on second partialFit call with different feature dimension', async () => {
        const som = new SOM({
          grid_width: 2, grid_height: 2, online_mode: true, random_state: 42,
        });
        const batch1 = tf.tensor2d([[0, 0], [1, 1]]);
        const batch2 = tf.tensor2d([[0, 0, 0], [1, 1, 1]]);

        await som.partial_fit(batch1);
        await expect(som.partial_fit(batch2)).rejects.toThrow('Feature dimension mismatch');

        batch1.dispose();
        batch2.dispose();
        som.dispose();
      });

      it('should include expected and actual dimensions in error message', async () => {
        const som = new SOM({
          grid_width: 2, grid_height: 2, online_mode: true, random_state: 42,
        });
        const batch1 = tf.tensor2d([[0, 0], [1, 1]]);
        const batch2 = tf.tensor2d([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]);

        await som.partial_fit(batch1);
        await expect(som.partial_fit(batch2)).rejects.toThrow(
          /expected 2.*got 5/
        );

        batch1.dispose();
        batch2.dispose();
        som.dispose();
      });

      it('should validate dimension when fit() was called first', async () => {
        const som = new SOM({
          grid_width: 2, grid_height: 2, online_mode: true,
          num_epochs: 5, random_state: 42,
        });
        const fit_data = tf.tensor2d([[0, 0], [1, 1]]);
        await som.fit(fit_data);

        const bad_batch = tf.tensor2d([[0, 0, 0], [1, 1, 1]]);
        await expect(som.partial_fit(bad_batch)).rejects.toThrow('Feature dimension mismatch');

        fit_data.dispose();
        bad_batch.dispose();
        som.dispose();
      });
    });

    describe('AC#5: getWeights() contract', () => {

      it('should throw if called before fit', () => {
        const som = new SOM({ grid_width: 2, grid_height: 2 });
        expect(() => som.get_weights()).toThrow('SOM must be fitted first');
        som.dispose();
      });

      it('should return correct shape [gridHeight][gridWidth][nFeatures]', async () => {
        const som = new SOM({
          grid_width: 4, grid_height: 3, num_epochs: 5, random_state: 42,
        });
        const X = tf.tensor2d([[0, 0], [1, 1]]);
        await som.fit(X);

        const weights = som.get_weights();
        expect(weights.length).toBe(3);         // gridHeight
        expect(weights[0].length).toBe(4);       // gridWidth
        expect(weights[0][0].length).toBe(2);    // nFeatures

        X.dispose();
        som.dispose();
      });

      it('should return a plain number[][][] array, not a tensor', async () => {
        const som = new SOM({
          grid_width: 2, grid_height: 2, num_epochs: 5, random_state: 42,
        });
        const X = tf.tensor2d([[0, 0], [1, 1]]);
        await som.fit(X);

        const weights = som.get_weights();
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
          grid_width: 2, grid_height: 2, online_mode: true,
          num_epochs: 5, random_state: 42,
        });
        const batch1 = tf.tensor2d([[0, 0], [1, 1]]);
        await som.partial_fit(batch1);

        const weights1 = som.get_weights();

        const batch2 = tf.tensor2d([[10, 10], [20, 20]]);
        await som.partial_fit(batch2);

        const weights2 = som.get_weights();

        // weights1 should be unchanged (snapshot)
        // weights2 should differ due to additional training
        let has_difference = false;
        for (let i = 0; i < 2; i++) {
          for (let j = 0; j < 2; j++) {
            for (let k = 0; k < 2; k++) {
              if (Math.abs(weights1[i][j][k] - weights2[i][j][k]) > 1e-6) {
                has_difference = true;
              }
            }
          }
        }
        expect(has_difference).toBe(true);

        batch1.dispose();
        batch2.dispose();
        som.dispose();
      });

      it('should be safe to use after dispose()', async () => {
        const som = new SOM({
          grid_width: 2, grid_height: 2, num_epochs: 5, random_state: 42,
        });
        const X = tf.tensor2d([[0, 0], [1, 1]]);
        await som.fit(X);

        const weights = som.get_weights();
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
          grid_width: 2, grid_height: 2, num_epochs: 5, random_state: 42,
        });
        const X = tf.tensor2d([[0, 0], [1, 1]]);
        await som.fit(X);

        const before_dispose = tf.memory().numTensors;
        som.dispose();
        const after_dispose = tf.memory().numTensors;

        expect(after_dispose).toBeLessThan(before_dispose);

        X.dispose();
      });

      it('should be safe to call dispose multiple times', async () => {
        const som = new SOM({
          grid_width: 2, grid_height: 2, num_epochs: 5, random_state: 42,
        });
        const X = tf.tensor2d([[0, 0], [1, 1]]);
        await som.fit(X);

        som.dispose();
        expect(() => som.dispose()).not.toThrow();

        X.dispose();
      });

      it('should cause getWeights to throw after dispose', async () => {
        const som = new SOM({
          grid_width: 2, grid_height: 2, num_epochs: 5, random_state: 42,
        });
        const X = tf.tensor2d([[0, 0], [1, 1]]);
        await som.fit(X);

        som.dispose();
        expect(() => som.get_weights()).toThrow('SOM must be fitted first');

        X.dispose();
      });
    });
  });
});