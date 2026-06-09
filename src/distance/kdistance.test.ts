import fs from 'fs';
import path from 'path';

import { kdistance } from './kdistance';

const FIXTURE_DIR = path.join(process.cwd(), '__fixtures__', 'density');

describe('kdistance', () => {
  it('returns the k-th nearest-neighbour distance (self counts first)', () => {
    // Points on a line at 0, 1, 3. Nearest-first distances (incl. self):
    //  p0: [0, 1, 3]   p1: [0, 1, 2]   p2: [0, 2, 3]
    const neighbor_distances = [
      [0, 1, 3],
      [0, 1, 2],
      [0, 2, 3],
    ];
    // k=1 -> self distance (0) for all points
    expect(Array.from(kdistance(neighbor_distances, 1))).toEqual([0, 0, 0]);
    // k=2 -> distance to nearest *other* neighbour
    expect(Array.from(kdistance(neighbor_distances, 2))).toEqual([1, 1, 2]);
    // k=3 -> distance to second-nearest other neighbour
    expect(Array.from(kdistance(neighbor_distances, 3))).toEqual([3, 2, 3]);
  });

  it('throws for non-positive or non-integer k', () => {
    expect(() => kdistance([[0, 1]], 0)).toThrow();
    expect(() => kdistance([[0, 1]], -1)).toThrow();
    expect(() => kdistance([[0, 1]], 1.5)).toThrow();
  });

  it('throws when a row has fewer than k neighbours', () => {
    expect(() => kdistance([[0, 1]], 3)).toThrow();
  });

  it('matches scipy-derived core distances on fixtures', () => {
    const files = fs
      .readdirSync(FIXTURE_DIR)
      .filter((f) => f.startsWith('kdistance_'));
    expect(files.length).toBeGreaterThan(0);

    for (const file of files) {
      const fixture = JSON.parse(
        fs.readFileSync(path.join(FIXTURE_DIR, file), 'utf-8'),
      ) as {
        neighbor_distances: number[][];
        k_values: number[];
        core_distances: Record<string, number[]>;
      };

      for (const k of fixture.k_values) {
        const got = Array.from(kdistance(fixture.neighbor_distances, k));
        const expected = fixture.core_distances[String(k)];
        expect(got.length).toBe(expected.length);
        for (let i = 0; i < expected.length; i++) {
          expect(got[i]).toBeCloseTo(expected[i], 10);
        }
      }
    }
  });
});
