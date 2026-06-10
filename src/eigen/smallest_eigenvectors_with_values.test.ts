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
   AC#1 + AC#2: both paths use the same near-zero tolerance
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
   AC#4: SpectralClustering n=99 (Jacobi) vs n=101 (Lanczos) path boundary
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
