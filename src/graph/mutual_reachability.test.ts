import fs from 'fs';
import path from 'path';

import { mutual_reachability } from './mutual_reachability';

const FIXTURE_DIR = path.join(process.cwd(), '__fixtures__', 'density');

describe('mutual_reachability', () => {
  it('computes max(core_i, core_j, d_ij) elementwise', () => {
    const D = [
      [0, 2, 5],
      [2, 0, 1],
      [5, 1, 0],
    ];
    const core = [3, 0.5, 4];
    const M = mutual_reachability(D, core);
    // M[0][1] = max(3, 0.5, 2) = 3
    expect(M[0][1]).toBe(3);
    // M[1][2] = max(0.5, 4, 1) = 4
    expect(M[1][2]).toBe(4);
    // M[0][2] = max(3, 4, 5) = 5
    expect(M[0][2]).toBe(5);
    // diagonal = max(core_i, core_i, 0) = core_i
    expect(M[0][0]).toBe(3);
    expect(M[1][1]).toBe(0.5);
    expect(M[1][0]).toBe(M[0][1]);
  });

  it('throws on dimension mismatch', () => {
    expect(() => mutual_reachability([[0, 1], [1, 0]], [1])).toThrow();
  });

  it('result is symmetric for asymmetric core distances', () => {
    const D = [
      [0, 2, 5],
      [2, 0, 1],
      [5, 1, 0],
    ];
    const core = [3, 0.5, 4];
    const M = mutual_reachability(D, core);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(M[i][j]).toBe(M[j][i]);
      }
    }
  });

  it('handles n=1 (single point)', () => {
    const M = mutual_reachability([[0]], [5]);
    expect(M).toHaveLength(1);
    expect(M[0][0]).toBe(5);
  });

  it('with all-zero core distances returns the distance matrix values', () => {
    const D = [[0, 3, 7], [3, 0, 2], [7, 2, 0]];
    const M = mutual_reachability(D, [0, 0, 0]);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(M[i][j]).toBe(D[i][j]);
      }
    }
  });

  it('accepts Float64Array as core_distances', () => {
    const D = [[0, 2, 5], [2, 0, 1], [5, 1, 0]];
    const core = [3, 0.5, 4];
    const M_arr = mutual_reachability(D, core);
    const M_typed = mutual_reachability(D, new Float64Array(core));
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(M_typed[i][j]).toBe(M_arr[i][j]);
      }
    }
  });

  it('throws on a non-square distance matrix', () => {
    expect(() =>
      mutual_reachability(
        [
          [0, 1, 2],
          [1, 0, 1],
        ],
        [1, 2],
      ),
    ).toThrow('square');
  });

  it('matches numpy-derived reachability on fixtures', () => {
    const files = fs
      .readdirSync(FIXTURE_DIR)
      .filter((f) => f.startsWith('mreach_'));
    expect(files.length).toBeGreaterThan(0);

    for (const file of files) {
      const fixture = JSON.parse(
        fs.readFileSync(path.join(FIXTURE_DIR, file), 'utf-8'),
      ) as {
        core_distances: number[];
        distance_matrix: number[][];
        mutual_reachability: number[][];
      };

      const M = mutual_reachability(
        fixture.distance_matrix,
        fixture.core_distances,
      );
      const expected = fixture.mutual_reachability;
      for (let i = 0; i < expected.length; i++) {
        for (let j = 0; j < expected.length; j++) {
          expect(M[i][j]).toBeCloseTo(expected[i][j], 10);
        }
      }
    }
  });
});
