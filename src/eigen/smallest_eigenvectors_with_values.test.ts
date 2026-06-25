import * as tf from '../backend/adapter';
import { smallest_eigenvectors_with_values } from './smallest_eigenvectors_with_values';
import { SpectralClustering } from '../clustering/spectral';

// Builds an n×n diagonal matrix from the given eigenvalue list.
function diagonal_matrix(eigenvalues: number[]): tf.Tensor2D {
  const n = eigenvalues.length;
  const flat = new Array(n * n).fill(0);
  for (let i = 0; i < n; i++) {
    flat[i * n + i] = eigenvalues[i];
  }
  return tf.tensor2d(flat, [n, n]);
}

// Builds data for two clearly separated clusters (inter-cluster distance 20,
// so RBF affinity between clusters ≈ exp(-200) ≈ 0 in float32).
// First Math.ceil(n/2) points sit near (0, 0); the rest near (20, 0).
function two_cluster_data(n: number): number[][] {
  const data: number[][] = [];
  const n1 = Math.ceil(n / 2);
  for (let i = 0; i < n1; i++) {
    data.push([i * 0.01, 0]);
  }
  for (let i = 0; i < n - n1; i++) {
    data.push([20 + i * 0.01, 0]);
  }
  return data;
}

/* =========================================================================
   Both eigensolver paths share the same near-zero tolerance, so path routing
   at the n=100 boundary cannot change the returned embedding dimension.
   ========================================================================= */

describe('smallest_eigenvectors_with_values – path tolerance alignment', () => {
  it('Jacobi path (n=4): eigenvalue 0.005 is not counted as near-zero', () => {
    // n=4 ≤ 100 → Jacobi path
    // eigenvalues: [0, 0.005, 1.0, 1.5]
    // With NEAR_ZERO_TOL=1e-5: c=1 (only exact 0), slice_cols = min(2+1, 4) = 3
    const M = diagonal_matrix([0, 0.005, 1.0, 1.5]);
    const result = smallest_eigenvectors_with_values(M, 2);

    expect(result.eigenvectors.shape[1]).toBe(3);

    result.eigenvectors.dispose();
    result.eigenvalues.dispose();
    M.dispose();
  });

  it('Lanczos path (n=105): eigenvalue 0.005 is not counted as near-zero', () => {
    // n=105 > 100 and k=2 < 105/3 ≈ 35 → Lanczos path
    // eigenvalues: [0, 0.005, 1.0, 1.01, ..., 2.02]
    // With NEAR_ZERO_TOL=1e-5: c=1 (only exact 0), slice_cols = min(2+1, 7) = 3
    const n = 105;
    const evals = [0, 0.005, ...Array.from({ length: n - 2 }, (_, i) => 1.0 + i * 0.01)];
    const M = diagonal_matrix(evals);
    const result = smallest_eigenvectors_with_values(M, 2);

    expect(result.eigenvectors.shape[1]).toBe(3);

    result.eigenvectors.dispose();
    result.eigenvalues.dispose();
    M.dispose();
  });

  it('both paths return the same eigenvector count for equivalent eigenvalue structure', () => {
    // Same leading eigenvalue pattern [0, 0.005, 1.0, …] on a small (Jacobi)
    // and large (Lanczos) matrix must produce the same slice_cols.
    const M_small = diagonal_matrix([0, 0.005, 1.0, 1.5]);

    const n = 105;
    const evals_large = [0, 0.005, ...Array.from({ length: n - 2 }, (_, i) => 1.0 + i * 0.01)];
    const M_large = diagonal_matrix(evals_large);

    const r_small = smallest_eigenvectors_with_values(M_small, 2);
    const r_large = smallest_eigenvectors_with_values(M_large, 2);

    expect(r_small.eigenvectors.shape[1]).toBe(r_large.eigenvectors.shape[1]);

    r_small.eigenvectors.dispose();
    r_small.eigenvalues.dispose();
    r_large.eigenvectors.dispose();
    r_large.eigenvalues.dispose();
    M_small.dispose();
    M_large.dispose();
  });
});

/* =========================================================================
   Input validation and returned eigenvalue content.
   ========================================================================= */

describe('smallest_eigenvectors_with_values – input validation', () => {
  it('rejects a non-positive or non-integer k', () => {
    const M = diagonal_matrix([0, 1, 2, 3]);
    expect(() => smallest_eigenvectors_with_values(M, 0)).toThrow(
      'k must be a positive integer',
    );
    expect(() => smallest_eigenvectors_with_values(M, -2)).toThrow(
      'k must be a positive integer',
    );
    expect(() => smallest_eigenvectors_with_values(M, 1.5)).toThrow(
      'k must be a positive integer',
    );
    M.dispose();
  });
});

describe('smallest_eigenvectors_with_values – eigenvalue content', () => {
  it('returns the smallest eigenvalues in ascending order (Jacobi path)', () => {
    // slice_cols = min(k + c, n) = min(2 + 1, 4) = 3, so the three smallest
    // eigenvalues of the diagonal come back ascending.
    const M = diagonal_matrix([1.5, 0, 1.0, 0.005]);
    const result = smallest_eigenvectors_with_values(M, 2);
    const vals = Array.from(result.eigenvalues.dataSync());
    expect(vals).toHaveLength(3);
    expect(vals[0]).toBeCloseTo(0, 5);
    expect(vals[1]).toBeCloseTo(0.005, 5);
    expect(vals[2]).toBeCloseTo(1.0, 5);
    result.eigenvectors.dispose();
    result.eigenvalues.dispose();
    M.dispose();
  });
});

/* =========================================================================
   Path equivalence for matrices with more than 5 structural zeros, where the
   k+5 Krylov buffer must expand to resolve the full degenerate zero eigenspace.
   ========================================================================= */

describe('smallest_eigenvectors_with_values – >5 near-zero eigenpairs', () => {
  // Builds an n×n block-diagonal complete-graph Laplacian with `num_blocks`
  // disconnected components, each a complete graph on `block_size` nodes.
  // Each component contributes exactly one zero eigenvalue (the block
  // indicator vector); all other eigenvalues equal `block_size`.
  // The large spectral gap (0 vs block_size) lets Lanczos converge quickly.
  function complete_graph_block_laplacian(block_size: number, num_blocks: number): tf.Tensor2D {
    const n = block_size * num_blocks;
    const flat = new Array(n * n).fill(0);
    for (let b = 0; b < num_blocks; b++) {
      const offset = b * block_size;
      for (let i = 0; i < block_size; i++) {
        for (let j = 0; j < block_size; j++) {
          flat[(offset + i) * n + (offset + j)] = i === j ? block_size - 1 : -1;
        }
      }
    }
    return tf.tensor2d(flat, [n, n]);
  }

  it.each([
    { k: 2, num_blocks: 6, jacobi_block_size: 2, lanczos_block_size: 17 },
    { k: 6, num_blocks: 6, jacobi_block_size: 2, lanczos_block_size: 17 },
    { k: 2, num_blocks: 8, jacobi_block_size: 2, lanczos_block_size: 13 },
  ])(
    'Lanczos (n=$lanczos_block_size×$num_blocks, k=$k, $num_blocks components) matches Jacobi column count',
    ({ k, num_blocks, jacobi_block_size, lanczos_block_size }) => {
      // Jacobi path: n = jacobi_block_size * num_blocks ≤ 100
      // Lanczos path: n = lanczos_block_size * num_blocks > 100, k < n/3
      // Both must return k + num_blocks columns — the near-zero count matches
      // the number of disconnected components.
      const expected_cols = k + num_blocks;

      const M_jacobi = complete_graph_block_laplacian(jacobi_block_size, num_blocks);
      const M_lanczos = complete_graph_block_laplacian(lanczos_block_size, num_blocks);

      const r_jacobi = smallest_eigenvectors_with_values(M_jacobi, k);
      const r_lanczos = smallest_eigenvectors_with_values(M_lanczos, k);

      expect(r_jacobi.eigenvectors.shape[1]).toBe(expected_cols);
      expect(r_lanczos.eigenvectors.shape[1]).toBe(expected_cols);

      r_jacobi.eigenvectors.dispose();
      r_jacobi.eigenvalues.dispose();
      r_lanczos.eigenvectors.dispose();
      r_lanczos.eigenvalues.dispose();
      M_jacobi.dispose();
      M_lanczos.dispose();
    },
  );
});

/* =========================================================================
   SpectralClustering yields the same labelling on either side of the n=100
   solver boundary: n=99 (Jacobi) vs n=101 (Lanczos).
   ========================================================================= */

describe('SpectralClustering – n=99 vs n=101 path boundary equivalence', () => {
  it.each([99, 101])(
    'correctly identifies 2 clusters for n=%i regardless of solver path',
    async (n) => {
      // n=99 → Jacobi path; n=101 → Lanczos path (n > 100 and k=2 < n/3)
      const data = two_cluster_data(n);
      const n1 = Math.ceil(n / 2);

      const model = new SpectralClustering({ n_clusters: 2 });
      await model.fit(data);
      const labels = model.labels_!;

      // All points in the first cluster share one label.
      const label_a = labels[0];
      for (let i = 1; i < n1; i++) {
        expect(labels[i]).toBe(label_a);
      }

      // All points in the second cluster share a different label.
      const label_b = labels[n1];
      expect(label_b).not.toBe(label_a);
      for (let i = n1 + 1; i < n; i++) {
        expect(labels[i]).toBe(label_b);
      }

      model.dispose();
    },
    30_000,
  );
});
