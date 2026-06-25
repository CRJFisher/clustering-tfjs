import fs from 'fs';
import path from 'path';

import { minimum_spanning_tree, MstEdge } from './minimum_spanning_tree';

const FIXTURE_DIR = path.join(process.cwd(), '__fixtures__', 'density');

function euclidean_matrix(X: number[][]): number[][] {
  const n = X.length;
  const D: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let s = 0;
      for (let d = 0; d < X[i].length; d++) {
        const diff = X[i][d] - X[j][d];
        s += diff * diff;
      }
      const dist = Math.sqrt(s);
      D[i][j] = dist;
      D[j][i] = dist;
    }
  }
  return D;
}

function sort_edges(edges: MstEdge[]): MstEdge[] {
  return [...edges].sort((a, b) =>
    a.source !== b.source ? a.source - b.source : a.target - b.target,
  );
}

describe('minimum_spanning_tree', () => {
  it('returns no edges for trivial graphs', () => {
    expect(minimum_spanning_tree([])).toEqual([]);
    expect(minimum_spanning_tree([[0]])).toEqual([]);
  });

  it('throws when the node count is invalid', () => {
    expect(() => minimum_spanning_tree(new Float64Array(0), -1)).toThrow(
      'valid node count',
    );
    expect(() =>
      minimum_spanning_tree(new Float64Array(4), Number.NaN),
    ).toThrow('valid node count');
  });

  it('builds the expected tree for a tiny line graph', () => {
    // Points at 0, 1, 2.5 -> MST connects 0-1 (1) and 1-2 (1.5)
    const D = euclidean_matrix([[0], [1], [2.5]]);
    const edges = sort_edges(minimum_spanning_tree(D));
    expect(edges.length).toBe(2);
    expect(edges[0]).toEqual({ source: 0, target: 1, weight: 1 });
    expect(edges[1]).toEqual({ source: 1, target: 2, weight: 1.5 });
  });

  it('accepts a flat Float64Array with explicit n', () => {
    const D = euclidean_matrix([[0], [1], [2.5]]);
    const flat = new Float64Array(9);
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) flat[i * 3 + j] = D[i][j];
    const edges = sort_edges(minimum_spanning_tree(flat, 3));
    expect(edges.length).toBe(2);
    expect(edges[0]).toEqual({ source: 0, target: 1, weight: 1 });
  });

  it('infers n from a flat Float64Array length when n is omitted', () => {
    const D = euclidean_matrix([[0], [1], [2.5]]);
    const flat = new Float64Array(9);
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) flat[i * 3 + j] = D[i][j];
    // round(sqrt(9)) = 3, so the tree matches the explicit-n result.
    const edges = sort_edges(minimum_spanning_tree(flat));
    expect(edges.length).toBe(2);
    expect(edges[0]).toEqual({ source: 0, target: 1, weight: 1 });
    expect(edges[1]).toEqual({ source: 1, target: 2, weight: 1.5 });
  });

  it('canonicalises edges so source < target when the lower-index node is absorbed last', () => {
    // D[0][2]=1 < D[0][1]=10, so node 2 is absorbed before node 1.
    // Node 1 is then absorbed via node 2 (best_source[1]=2 > u=1), triggering
    // the a > u branch of the canonicalisation.
    const D = [
      [0, 10, 1],
      [10, 0, 2],
      [1, 2, 0],
    ];
    const edges = sort_edges(minimum_spanning_tree(D));
    expect(edges.length).toBe(2);
    expect(edges[0]).toEqual({ source: 0, target: 2, weight: 1 });
    expect(edges[1]).toEqual({ source: 1, target: 2, weight: 2 });
    // Verify the invariant directly.
    for (const e of edges) expect(e.source).toBeLessThan(e.target);
  });

  it('throws on a disconnected (infinite-weight) graph', () => {
    const inf = Number.POSITIVE_INFINITY;
    const D = [
      [0, 1, inf],
      [1, 0, inf],
      [inf, inf, 0],
    ];
    expect(() => minimum_spanning_tree(D)).toThrow();
  });

  it('matches the scipy MST edge set on fixtures', () => {
    const files = fs
      .readdirSync(FIXTURE_DIR)
      .filter((f) => f.startsWith('mst_'));
    expect(files.length).toBeGreaterThan(0);

    for (const file of files) {
      const fixture = JSON.parse(
        fs.readFileSync(path.join(FIXTURE_DIR, file), 'utf-8'),
      ) as { X: number[][]; edges: number[][]; total_weight: number };

      const D = euclidean_matrix(fixture.X);
      const edges = sort_edges(minimum_spanning_tree(D));

      expect(edges.length).toBe(fixture.edges.length);
      for (let i = 0; i < edges.length; i++) {
        expect(edges[i].source).toBe(fixture.edges[i][0]);
        expect(edges[i].target).toBe(fixture.edges[i][1]);
        expect(edges[i].weight).toBeCloseTo(fixture.edges[i][2], 9);
      }

      const total = edges.reduce((s, e) => s + e.weight, 0);
      expect(total).toBeCloseTo(fixture.total_weight, 9);
    }
  });
});
