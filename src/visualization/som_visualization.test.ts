import * as tf from '../../test_support/tensorflow_helper';
import { SOM } from '../clustering/som';
import {
  get_component_planes,
  get_hit_map,
  get_activation_map,
  track_bmu_trajectory,
  get_quantization_quality_map,
  get_neighbor_distance_matrix,
  export_for_visualization,
  get_density_map,
} from './som_visualization';

describe('som_visualization', () => {
  let som: SOM;
  let X: tf.Tensor2D;
  const grid_width = 3, grid_height = 3;
  const train_data = [[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5], [0.2, 0.8]];

  beforeAll(async () => {
    som = new SOM({
      grid_width,
      grid_height,
      num_epochs: 10,
      random_state: 42,
      learning_rate: 0.5,
      radius: 1.5,
    });
    X = tf.tensor2d(train_data);
    await som.fit(X);
  });

  afterAll(() => {
    X.dispose();
    som.dispose();
  });

  describe('get_component_planes', () => {
    it('returns shape [2, 3, 3] for 2 features', () => {
      const result = get_component_planes(som);
      try {
        expect(result.shape).toEqual([2, grid_height, grid_width]);
      } finally {
        result.dispose();
      }
    });

    it('contains all finite values', () => {
      const result = get_component_planes(som);
      try {
        const values = result.dataSync();
        for (const v of values) {
          expect(Number.isFinite(v)).toBe(true);
        }
      } finally {
        result.dispose();
      }
    });
  });

  describe('get_hit_map', () => {
    it('returns shape [3, 3]', async () => {
      const result = await get_hit_map(som, X);
      try {
        expect(result.shape).toEqual([grid_height, grid_width]);
      } finally {
        result.dispose();
      }
    });

    it('has total hits equal to the number of samples', async () => {
      const result = await get_hit_map(som, X);
      try {
        const values = result.dataSync();
        const total_hits = Array.from(values).reduce((sum, v) => sum + v, 0);
        expect(total_hits).toBe(train_data.length);
      } finally {
        result.dispose();
      }
    });

    it('has all non-negative values', async () => {
      const result = await get_hit_map(som, X);
      try {
        const values = result.dataSync();
        for (const v of values) {
          expect(v).toBeGreaterThanOrEqual(0);
        }
      } finally {
        result.dispose();
      }
    });
  });

  describe('get_activation_map', () => {
    it('returns shape [3, 3]', () => {
      const sample = tf.tensor1d([0.5, 0.5]);
      try {
        const result = get_activation_map(som, sample);
        try {
          expect(result.shape).toEqual([grid_height, grid_width]);
        } finally {
          result.dispose();
        }
      } finally {
        sample.dispose();
      }
    });

    it('gives the BMU the highest activation', () => {
      const sample = tf.tensor1d([0.5, 0.5]);
      try {
        const result = get_activation_map(som, sample);
        try {
          const values = Array.from(result.dataSync());
          const max_val = Math.max(...values);
          // Activation is (max_dist - dist) / max_dist, so BMU gets the highest value
          expect(max_val).toBeGreaterThan(0.5);
          const max_count = values.filter(v => Math.abs(v - max_val) < 1e-6).length;
          expect(max_count).toBeGreaterThanOrEqual(1);
        } finally {
          result.dispose();
        }
      } finally {
        sample.dispose();
      }
    });

    it('has all values finite and in [0, 1]', () => {
      const sample = tf.tensor1d([0.5, 0.5]);
      try {
        const result = get_activation_map(som, sample);
        try {
          const values = result.dataSync();
          for (const v of values) {
            expect(Number.isFinite(v)).toBe(true);
            expect(v).toBeGreaterThanOrEqual(0);
            expect(v).toBeLessThanOrEqual(1);
          }
        } finally {
          result.dispose();
        }
      } finally {
        sample.dispose();
      }
    });
  });

  describe('track_bmu_trajectory', () => {
    it('returns output length matching input length', async () => {
      const sequence = tf.tensor2d([[0, 0], [0.5, 0.5], [1, 1]]);
      try {
        const trajectory = await track_bmu_trajectory(som, sequence);
        expect(trajectory.length).toBe(3);
      } finally {
        sequence.dispose();
      }
    });

    it('returns [row, col] entries within grid bounds', async () => {
      const sequence = tf.tensor2d([[0, 0], [0.5, 0.5], [1, 1]]);
      try {
        const trajectory = await track_bmu_trajectory(som, sequence);
        for (const entry of trajectory) {
          expect(entry).toHaveLength(2);
          const [row, col] = entry;
          expect(row).toBeGreaterThanOrEqual(0);
          expect(row).toBeLessThan(grid_height);
          expect(col).toBeGreaterThanOrEqual(0);
          expect(col).toBeLessThan(grid_width);
        }
      } finally {
        sequence.dispose();
      }
    });
  });

  describe('get_quantization_quality_map', () => {
    it('returns shape [3, 3]', async () => {
      const result = await get_quantization_quality_map(som, X);
      try {
        expect(result.shape).toEqual([grid_height, grid_width]);
      } finally {
        result.dispose();
      }
    });

    it('has all non-negative values', async () => {
      const result = await get_quantization_quality_map(som, X);
      try {
        const values = result.dataSync();
        for (const v of values) {
          expect(v).toBeGreaterThanOrEqual(0);
        }
      } finally {
        result.dispose();
      }
    });
  });

  describe('get_neighbor_distance_matrix', () => {
    it('returns shape [3, 3]', () => {
      const result = get_neighbor_distance_matrix(som);
      try {
        expect(result.shape).toEqual([grid_height, grid_width]);
      } finally {
        result.dispose();
      }
    });

    it('has all non-negative and finite values', () => {
      const result = get_neighbor_distance_matrix(som);
      try {
        const values = result.dataSync();
        for (const v of values) {
          expect(v).toBeGreaterThanOrEqual(0);
          expect(Number.isFinite(v)).toBe(true);
        }
      } finally {
        result.dispose();
      }
    });
  });

  describe('export_for_visualization', () => {
    it('produces JSON with expected keys', async () => {
      const json_str = await export_for_visualization(som, 'json');
      const parsed = JSON.parse(json_str);
      expect(parsed).toHaveProperty('grid_height');
      expect(parsed).toHaveProperty('grid_width');
      expect(parsed).toHaveProperty('weights');
      expect(parsed).toHaveProperty('u_matrix');
      expect(parsed).toHaveProperty('params');
    });

    it('produces CSV with correct header and data rows', async () => {
      const csv_str = await export_for_visualization(som, 'csv');
      const lines = csv_str.trim().split('\n');
      const header = lines[0];
      expect(header.length).toBeGreaterThan(0);
      const data_rows = lines.slice(1);
      expect(data_rows.length).toBe(grid_height * grid_width);
    });
  });

  describe('get_density_map', () => {
    it('returns shape [3, 3]', async () => {
      const result = await get_density_map(som, X);
      try {
        expect(result.shape).toEqual([grid_height, grid_width]);
      } finally {
        result.dispose();
      }
    });

    it('has all non-negative values', async () => {
      const result = await get_density_map(som, X);
      try {
        const values = result.dataSync();
        for (const v of values) {
          expect(v).toBeGreaterThanOrEqual(0);
        }
      } finally {
        result.dispose();
      }
    });

    it('returns raw hit map values when sigma <= 0', async () => {
      const density_result = await get_density_map(som, X, 0);
      const hit_result = await get_hit_map(som, X);
      try {
        const density_values = Array.from(density_result.dataSync());
        const hit_values = Array.from(hit_result.dataSync());
        for (let i = 0; i < density_values.length; i++) {
          expect(density_values[i]).toBeCloseTo(hit_values[i]);
        }
      } finally {
        density_result.dispose();
        hit_result.dispose();
      }
    });
  });
});
