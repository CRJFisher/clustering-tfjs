import * as tf from '../tensorflow-helper';
import { SOM } from '../../src/clustering/som';
import {
  getComponentPlanes,
  getHitMap,
  getActivationMap,
  trackBMUTrajectory,
  getQuantizationQualityMap,
  getNeighborDistanceMatrix,
  exportForVisualization,
  getDensityMap,
} from '../../src/utils/som_visualization';

describe('SOM Visualization Utilities', () => {
  let som: SOM;
  let X: tf.Tensor2D;
  const gridWidth = 3, gridHeight = 3;
  const trainData = [[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5], [0.2, 0.8]];

  beforeAll(async () => {
    som = new SOM({
      gridWidth,
      gridHeight,
      numEpochs: 10,
      randomState: 42,
      learningRate: 0.5,
      radius: 1.5,
    });
    X = tf.tensor2d(trainData);
    await som.fit(X);
  });

  afterAll(() => {
    X.dispose();
    som.dispose();
  });

  describe('getComponentPlanes', () => {
    it('should return shape [2, 3, 3] for 2 features', () => {
      const result = getComponentPlanes(som);
      try {
        expect(result.shape).toEqual([2, gridHeight, gridWidth]);
      } finally {
        result.dispose();
      }
    });

    it('should contain all finite values', () => {
      const result = getComponentPlanes(som);
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

  describe('getHitMap', () => {
    it('should return shape [3, 3]', async () => {
      const result = await getHitMap(som, X);
      try {
        expect(result.shape).toEqual([gridHeight, gridWidth]);
      } finally {
        result.dispose();
      }
    });

    it('should have total hits equal to number of samples', async () => {
      const result = await getHitMap(som, X);
      try {
        const values = result.dataSync();
        const totalHits = Array.from(values).reduce((sum, v) => sum + v, 0);
        expect(totalHits).toBe(trainData.length);
      } finally {
        result.dispose();
      }
    });

    it('should have all non-negative values', async () => {
      const result = await getHitMap(som, X);
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

  describe('getActivationMap', () => {
    it('should return shape [3, 3]', () => {
      const sample = tf.tensor1d([0.5, 0.5]);
      try {
        const result = getActivationMap(som, sample);
        try {
          expect(result.shape).toEqual([gridHeight, gridWidth]);
        } finally {
          result.dispose();
        }
      } finally {
        sample.dispose();
      }
    });

    it('should have BMU with highest activation', () => {
      const sample = tf.tensor1d([0.5, 0.5]);
      try {
        const result = getActivationMap(som, sample);
        try {
          const values = Array.from(result.dataSync());
          const maxVal = Math.max(...values);
          // Activation is (maxDist - dist) / maxDist, so BMU gets the highest value
          expect(maxVal).toBeGreaterThan(0.5);
          // There should be exactly one maximum
          const maxCount = values.filter(v => Math.abs(v - maxVal) < 1e-6).length;
          expect(maxCount).toBeGreaterThanOrEqual(1);
        } finally {
          result.dispose();
        }
      } finally {
        sample.dispose();
      }
    });

    it('should have all values finite and in [0, 1]', () => {
      const sample = tf.tensor1d([0.5, 0.5]);
      try {
        const result = getActivationMap(som, sample);
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

  describe('trackBMUTrajectory', () => {
    it('should return output length matching input length', async () => {
      const sequence = tf.tensor2d([[0, 0], [0.5, 0.5], [1, 1]]);
      try {
        const trajectory = await trackBMUTrajectory(som, sequence);
        expect(trajectory.length).toBe(3);
      } finally {
        sequence.dispose();
      }
    });

    it('should return [row, col] entries within grid bounds', async () => {
      const sequence = tf.tensor2d([[0, 0], [0.5, 0.5], [1, 1]]);
      try {
        const trajectory = await trackBMUTrajectory(som, sequence);
        for (const entry of trajectory) {
          expect(entry).toHaveLength(2);
          const [row, col] = entry;
          expect(row).toBeGreaterThanOrEqual(0);
          expect(row).toBeLessThan(gridHeight);
          expect(col).toBeGreaterThanOrEqual(0);
          expect(col).toBeLessThan(gridWidth);
        }
      } finally {
        sequence.dispose();
      }
    });
  });

  describe('getQuantizationQualityMap', () => {
    it('should return shape [3, 3]', async () => {
      const result = await getQuantizationQualityMap(som, X);
      try {
        expect(result.shape).toEqual([gridHeight, gridWidth]);
      } finally {
        result.dispose();
      }
    });

    it('should have all non-negative values', async () => {
      const result = await getQuantizationQualityMap(som, X);
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

  describe('getNeighborDistanceMatrix', () => {
    it('should return shape [3, 3]', () => {
      const result = getNeighborDistanceMatrix(som);
      try {
        expect(result.shape).toEqual([gridHeight, gridWidth]);
      } finally {
        result.dispose();
      }
    });

    it('should have all non-negative and finite values', () => {
      const result = getNeighborDistanceMatrix(som);
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

  describe('exportForVisualization', () => {
    it('should produce JSON with expected keys', async () => {
      const jsonStr = await exportForVisualization(som, 'json');
      const parsed = JSON.parse(jsonStr);
      expect(parsed).toHaveProperty('gridHeight');
      expect(parsed).toHaveProperty('gridWidth');
      expect(parsed).toHaveProperty('weights');
      expect(parsed).toHaveProperty('uMatrix');
      expect(parsed).toHaveProperty('params');
    });

    it('should produce CSV with correct header and data rows', async () => {
      const csvStr = await exportForVisualization(som, 'csv');
      const lines = csvStr.trim().split('\n');
      // First line is the header
      const header = lines[0];
      expect(header.length).toBeGreaterThan(0);
      // Remaining lines are data rows, one per grid cell
      const dataRows = lines.slice(1);
      expect(dataRows.length).toBe(gridHeight * gridWidth);
    });
  });

  describe('getDensityMap', () => {
    it('should return shape [3, 3]', async () => {
      const result = await getDensityMap(som, X);
      try {
        expect(result.shape).toEqual([gridHeight, gridWidth]);
      } finally {
        result.dispose();
      }
    });

    it('should have all non-negative values', async () => {
      const result = await getDensityMap(som, X);
      try {
        const values = result.dataSync();
        for (const v of values) {
          expect(v).toBeGreaterThanOrEqual(0);
        }
      } finally {
        result.dispose();
      }
    });

    it('should return raw hit map values when sigma <= 0', async () => {
      const densityResult = await getDensityMap(som, X, 0);
      const hitResult = await getHitMap(som, X);
      try {
        const densityValues = Array.from(densityResult.dataSync());
        const hitValues = Array.from(hitResult.dataSync());
        for (let i = 0; i < densityValues.length; i++) {
          expect(densityValues[i]).toBeCloseTo(hitValues[i]);
        }
      } finally {
        densityResult.dispose();
        hitResult.dispose();
      }
    });
  });
});
