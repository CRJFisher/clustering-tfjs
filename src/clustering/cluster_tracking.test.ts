import fs from 'fs';
import path from 'path';

import { track_clusters, type ClusterTransition } from '..';

const FIXTURE_DIR = path.join(process.cwd(), '__fixtures__', 'tracking');

interface TrackingFixture {
  threshold: number;
  prev_centroids: number[][];
  curr_centroids: number[][];
  cost_matrix: number[][];
  assignment: number[];
  transitions: { type: string; prev: number[]; curr: number[] }[];
}

/** Canonical, order-independent key for a transition. */
function transition_key(t: { type: string; prev: number[]; curr: number[] }): string {
  const p = [...t.prev].sort((a, b) => a - b).join(',');
  const c = [...t.curr].sort((a, b) => a - b).join(',');
  return `${t.type}|${p}|${c}`;
}

describe('track_clusters – reference parity (scipy linear_sum_assignment)', () => {
  const files = fs.readdirSync(FIXTURE_DIR).filter((f) => f.endsWith('.json'));

  for (const file of files) {
    const fixture = JSON.parse(
      fs.readFileSync(path.join(FIXTURE_DIR, file), 'utf-8'),
    ) as TrackingFixture;

    it(`matches assignment and transitions for ${file}`, () => {
      const result = track_clusters(
        fixture.prev_centroids,
        fixture.curr_centroids,
        { threshold: fixture.threshold },
      );

      // Cost matrix matches scipy cosine distances.
      for (let i = 0; i < fixture.cost_matrix.length; i++) {
        for (let j = 0; j < fixture.cost_matrix[i].length; j++) {
          expect(result.cost_matrix[i][j]).toBeCloseTo(
            fixture.cost_matrix[i][j],
            4,
          );
        }
      }

      // Pruned per-current assignment matches the reference.
      expect(result.assignment).toEqual(fixture.assignment);

      // Transition set matches the reference (order-independent).
      const mine = new Set(result.transitions.map(transition_key));
      const ref = new Set(fixture.transitions.map(transition_key));
      expect(mine).toEqual(ref);
    });
  }
});

describe('track_clusters – behaviour', () => {
  it('keeps lifeline ids stable for persisting clusters across frames', () => {
    const a = [
      [1, 0],
      [0, 1],
    ];
    const b = [
      [0.95, 0.05],
      [0.05, 0.95],
    ];
    const c = [
      [0.9, 0.1],
      [0.1, 0.9],
    ];

    const r1 = track_clusters(a, b, { threshold: 0.5 });
    // Persisting clusters keep their seeded lifelines (0,1) permuted by match.
    const r2 = track_clusters(b, c, { threshold: 0.5 }, r1.state);
    // The cluster that persisted should carry forward r1's lifeline id.
    for (let j = 0; j < c.length; j++) {
      const i = r2.assignment[j];
      if (i >= 0) {
        expect(r2.state.lifelines[j]).toBe(r1.state.lifelines[i]);
      }
    }
  });

  it('assigns fresh lifeline ids to emerging clusters', () => {
    const prev = [[1, 0]];
    const curr = [
      [1, 0],
      [0, 1], // brand new direction -> EMERGE
    ];
    const result = track_clusters(prev, curr, { threshold: 0.5 });
    const types = result.transitions.map((t) => t.type);
    expect(types).toContain('EMERGE');
    // Emerged lifeline differs from the seeded prev lifeline (0).
    const emerge = result.transitions.find(
      (t: ClusterTransition) => t.type === 'EMERGE',
    )!;
    expect(emerge.lifeline_id).toBeGreaterThanOrEqual(prev.length);
  });

  it('emits DIE when a cluster has no current match', () => {
    const prev = [
      [1, 0],
      [0, 1],
    ];
    const curr = [[1, 0]];
    const result = track_clusters(prev, curr, { threshold: 0.5 });
    expect(result.transitions.map((t) => t.type)).toContain('DIE');
  });

  it('is stateless: repeated calls give identical results', () => {
    const prev = [
      [1, 0],
      [0, 1],
    ];
    const curr = [
      [0.9, 0.1],
      [0.1, 0.9],
    ];
    const r1 = track_clusters(prev, curr, { threshold: 0.5 });
    const r2 = track_clusters(prev, curr, { threshold: 0.5 });
    expect(r1.assignment).toEqual(r2.assignment);
    expect(r1.transitions.map(transition_key).sort()).toEqual(
      r2.transitions.map(transition_key).sort(),
    );
  });
});
