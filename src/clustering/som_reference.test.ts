/**
 * SOM numeric reference suite: batch-equivalence against MiniSom.
 *
 * Each fixture pins MiniSom's deterministic `train_batch` output starting from an
 * injected initial weight grid. `train_minisom_reference` replicates that exact
 * algorithm, so weights, BMUs, labels, quantization/topographic error, and the
 * U-matrix all match the reference to floating-point precision.
 *
 * The production online mini-batch training path (`SOM.fit`) is a different,
 * GPU-friendly algorithm and is NOT exercised here; its properties are validated
 * by `som.test.ts` and `som_hexagonal.test.ts`. Regenerate fixtures with
 * `tools/sklearn_fixtures/generate_som.py` (see `docs/som-benchmarking.md`).
 */
import * as fs from 'fs';
import * as path from 'path';
import { train_minisom_reference } from './som_reference_training';
import type { SOMTopology, SOMNeighborhood } from './types';

interface ReferenceFixture {
  name: string;
  X: number[][];
  params: {
    grid_width: number;
    grid_height: number;
    topology: SOMTopology;
    neighborhood: SOMNeighborhood;
    learning_rate: number;
    radius: number;
    num_iteration: number;
    random_state: number;
  };
  initial_weights: number[][][];
  weights: number[][][];
  bmus: number[][];
  labels: number[];
  u_matrix: number[][];
  metrics: { quantization_error: number; topographic_error: number };
}

// Per-element weight tolerance. The reference trainer is a literal transcription
// of MiniSom's float64 arithmetic, so observed parity is ~1e-12. We assert 1e-9
// to leave headroom for benign V8 floating-point reordering while staying six
// orders of magnitude tighter than the AC ceiling of 1e-3.
const WEIGHT_TOL = 1e-9;
// Quantization error and the U-matrix are continuous functions of the matched
// weights, so they also match to ~1e-12; we assert 1e-9.
const QE_TOL = 1e-9;
const U_MATRIX_TOL = 1e-9;
// Topographic error is a DISCRETE metric: it counts samples whose two nearest
// neurons are not grid-adjacent. Selecting the second-nearest neuron is
// discontinuous at distance degeneracies, where a sub-1e-9 weight difference
// flips one sample's classification. Under matched weights it agrees with
// MiniSom on all but rare near-degenerate samples, so we hold the AC's stated
// <=1% relative bound (with an exact-zero guard) rather than a float tolerance.
const TE_REL_TOL = 0.01;

const fixtures_dir = path.join(process.cwd(), '__fixtures__', 'som');
const fixtures: ReferenceFixture[] = fs
  .readdirSync(fixtures_dir)
  .filter((f) => f.endsWith('.json'))
  .map((file) => JSON.parse(fs.readFileSync(path.join(fixtures_dir, file), 'utf8')));

function max_abs_diff_3d(a: number[][][], b: number[][][]): number {
  let max = 0;
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a[i].length; j++) {
      for (let k = 0; k < a[i][j].length; k++) {
        max = Math.max(max, Math.abs(a[i][j][k] - b[i][j][k]));
      }
    }
  }
  return max;
}

function max_abs_diff_2d(a: number[][], b: number[][]): number {
  let max = 0;
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a[i].length; j++) {
      max = Math.max(max, Math.abs(a[i][j] - b[i][j]));
    }
  }
  return max;
}

describe('SOM numeric reference (MiniSom train_batch equivalence)', () => {
  expect(fixtures.length).toBeGreaterThan(0);

  for (const fixture of fixtures) {
    describe(fixture.name, () => {
      const result = train_minisom_reference(fixture.X, fixture.initial_weights, {
        grid_width: fixture.params.grid_width,
        grid_height: fixture.params.grid_height,
        topology: fixture.params.topology,
        neighborhood: fixture.params.neighborhood,
        learning_rate: fixture.params.learning_rate,
        sigma: fixture.params.radius,
        num_iteration: fixture.params.num_iteration,
      });

      it('matches MiniSom weights per element', () => {
        expect(result.weights.length).toBe(fixture.weights.length);
        expect(result.weights[0].length).toBe(fixture.weights[0].length);
        expect(result.weights[0][0].length).toBe(fixture.weights[0][0].length);
        expect(max_abs_diff_3d(result.weights, fixture.weights)).toBeLessThanOrEqual(WEIGHT_TOL);
      });

      it('matches MiniSom quantization error', () => {
        expect(Math.abs(result.quantization_error - fixture.metrics.quantization_error)).toBeLessThanOrEqual(
          QE_TOL,
        );
      });

      it('matches MiniSom topographic error', () => {
        const reference_te = fixture.metrics.topographic_error;
        if (reference_te === 0) {
          expect(result.topographic_error).toBe(0);
        } else {
          expect(Math.abs(result.topographic_error - reference_te) / reference_te).toBeLessThanOrEqual(
            TE_REL_TOL,
          );
        }
      });

      it('matches MiniSom BMU indices exactly', () => {
        expect(result.bmus).toEqual(fixture.bmus);
      });

      it('matches MiniSom cluster labels exactly', () => {
        expect(result.labels).toEqual(fixture.labels);
      });

      it('matches MiniSom U-matrix per element', () => {
        expect(result.u_matrix.length).toBe(fixture.u_matrix.length);
        expect(result.u_matrix[0].length).toBe(fixture.u_matrix[0].length);
        expect(max_abs_diff_2d(result.u_matrix, fixture.u_matrix)).toBeLessThanOrEqual(U_MATRIX_TOL);
      });
    });
  }
});
