import * as tf from '../backend/adapter';
import { lanczos_smallest_eigenpairs } from './lanczos';
import { improved_jacobi_eigen } from './improved';
import { normalised_laplacian } from '../graph/laplacian';
import { make_random_stream } from '../random';

describe('Lanczos eigensolver', () => {

  describe('basic correctness', () => {
    it('finds eigenvalues of a diagonal matrix', () => {
      const diag = [3, 1, 4, 2, 5];
      const n = diag.length;
      const matrix = tf.tensor2d(
        Array.from({ length: n }, (_, i) =>
          Array.from({ length: n }, (_, j) => (i === j ? diag[i] : 0)),
        ),
      );

      const result = lanczos_smallest_eigenpairs(matrix, 3, { random_seed: 42 });
      matrix.dispose();

      // Expect the 3 smallest: [1, 2, 3] (no degenerate eigenvalues)
      expect(result.eigenvalues).toHaveLength(3);
      expect(result.eigenvalues[0]).toBeCloseTo(1, 4);
      expect(result.eigenvalues[1]).toBeCloseTo(2, 4);
      expect(result.eigenvalues[2]).toBeCloseTo(3, 4);
    });

    it('finds eigenvalues of a known symmetric matrix', () => {
      // 2x2 symmetric matrix [[2, 1], [1, 2]]
      // Eigenvalues: 1 and 3
      const matrix = tf.tensor2d([[2, 1], [1, 2]]);

      const result = lanczos_smallest_eigenpairs(matrix, 1, { random_seed: 42, is_psd: false });
      matrix.dispose();

      expect(result.eigenvalues).toHaveLength(1);
      expect(result.eigenvalues[0]).toBeCloseTo(1, 5);
    });

    it('finds eigenvalues through a matrix-free operator', () => {
      const diag = [3, 1, 4, 2, 5];
      const operator = {
        n: diag.length,
        matvec(vector: Float64Array): Float64Array {
          const result = new Float64Array(diag.length);
          for (let i = 0; i < diag.length; i++) {
            result[i] = diag[i] * vector[i];
          }
          return result;
        },
      };

      const result = lanczos_smallest_eigenpairs(operator, 3, {
        random_seed: 42,
      });

      expect(result.eigenvalues).toHaveLength(3);
      expect(result.eigenvalues[0]).toBeCloseTo(1, 4);
      expect(result.eigenvalues[1]).toBeCloseTo(2, 4);
      expect(result.eigenvalues[2]).toBeCloseTo(3, 4);
    });

    it('finds all eigenvalues when k equals n', () => {
      const matrix = tf.tensor2d([
        [4, 1, 0],
        [1, 3, 1],
        [0, 1, 2],
      ]);

      const result = lanczos_smallest_eigenpairs(matrix, 3, { random_seed: 42, is_psd: false });
      matrix.dispose();

      // Compare with Jacobi
      const matrix_again = tf.tensor2d([
        [4, 1, 0],
        [1, 3, 1],
        [0, 1, 2],
      ]);
      const jacobi = improved_jacobi_eigen(matrix_again);
      matrix_again.dispose();

      expect(result.eigenvalues).toHaveLength(3);
      for (let i = 0; i < 3; i++) {
        expect(result.eigenvalues[i]).toBeCloseTo(jacobi.eigenvalues[i], 3);
      }
    });
  });

  describe('normalized Laplacian properties', () => {
    it('finds near-zero eigenvalues for connected graph Laplacian', () => {
      // Build a simple 4-node connected graph's normalized Laplacian
      // I - D^{-1/2} A D^{-1/2}
      const A = tf.tensor2d([
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [0, 1, 1, 0],
      ]);

      const L = normalised_laplacian(A) as tf.Tensor2D;
      A.dispose();

      const result = lanczos_smallest_eigenpairs(L, 2, {
        random_seed: 42,
        is_psd: true,
      });
      L.dispose();

      // Connected graph: smallest eigenvalue ≈ 0
      expect(result.eigenvalues[0]).toBeCloseTo(0, 3);
      // Second eigenvalue should be positive (Fiedler value)
      expect(result.eigenvalues[1]).toBeGreaterThan(0.01);
    });

    it('finds multiple zero eigenvalues for disconnected graph', () => {
      // Two disconnected components: {0,1} and {2,3}
      const A = tf.tensor2d([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
      ]);

      const L = normalised_laplacian(A) as tf.Tensor2D;
      A.dispose();

      const result = lanczos_smallest_eigenpairs(L, 3, {
        random_seed: 42,
        is_psd: true,
      });
      L.dispose();

      // Two components → two zero eigenvalues
      expect(result.eigenvalues[0]).toBeCloseTo(0, 3);
      expect(result.eigenvalues[1]).toBeCloseTo(0, 3);
      expect(result.eigenvalues[2]).toBeGreaterThan(0.1);
    });
  });

  describe('parity with Jacobi solver', () => {
    it('produces equivalent eigenvectors for medium matrix (n=50)', () => {
      // Generate a random symmetric PSD matrix
      const n = 50;
      const k = 5;

      // Create a random symmetric matrix via A = Q * diag * Q^T + I
      const rng = make_random_stream(42);
      const raw: number[][] = Array.from({ length: n }, () =>
        Array.from({ length: n }, () => rng.rand() - 0.5),
      );

      // Make symmetric
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          raw[i][j] = raw[j][i] = (raw[i][j] + raw[j][i]) / 2;
        }
      }
      // Make PSD by adding n*I
      for (let i = 0; i < n; i++) raw[i][i] += n;

      const matrix = tf.tensor2d(raw);

      const lanczos_result = lanczos_smallest_eigenpairs(matrix, k, {
        random_seed: 42,
        is_psd: true,
      });

      const jacobi_result = improved_jacobi_eigen(matrix, {
        is_psd: true,
        max_iterations: 3000,
        tolerance: 1e-14,
      });
      matrix.dispose();

      // Compare eigenvalues (Jacobi returns all n, sorted ascending)
      for (let i = 0; i < k; i++) {
        expect(lanczos_result.eigenvalues[i]).toBeCloseTo(
          jacobi_result.eigenvalues[i],
          2,
        );
      }

      // Compare eigenvectors via |dot product| ≈ 1 for non-degenerate eigenvalues
      for (let col = 0; col < k; col++) {
        let dot = 0;
        for (let row = 0; row < n; row++) {
          dot +=
            lanczos_result.eigenvectors[row][col] *
            jacobi_result.eigenvectors[row][col];
        }
        // Allow for sign flip
        expect(Math.abs(dot)).toBeGreaterThan(0.9);
      }
    });
  });

  describe('edge cases', () => {
    it('throws for k < 1', () => {
      const matrix = tf.tensor2d([[1, 0], [0, 1]]);
      expect(() => lanczos_smallest_eigenpairs(matrix, 0)).toThrow();
      matrix.dispose();
    });

    it('throws for k > n', () => {
      const matrix = tf.tensor2d([[1, 0], [0, 1]]);
      expect(() => lanczos_smallest_eigenpairs(matrix, 3)).toThrow();
      matrix.dispose();
    });

    it('handles identity matrix', () => {
      const n = 10;
      const matrix = tf.eye(n) as tf.Tensor2D;
      const result = lanczos_smallest_eigenpairs(matrix, 3, { random_seed: 42 });
      matrix.dispose();

      // All eigenvalues should be 1
      for (let i = 0; i < 3; i++) {
        expect(result.eigenvalues[i]).toBeCloseTo(1, 3);
      }
    });

    it('is deterministic with same seed', () => {
      const matrix = tf.tensor2d([
        [4, 1, 0, 0],
        [1, 3, 1, 0],
        [0, 1, 2, 1],
        [0, 0, 1, 1],
      ]);

      const r1 = lanczos_smallest_eigenpairs(matrix, 2, { random_seed: 42, is_psd: false });
      const r2 = lanczos_smallest_eigenpairs(matrix, 2, { random_seed: 42, is_psd: false });
      matrix.dispose();

      for (let i = 0; i < 2; i++) {
        expect(r1.eigenvalues[i]).toEqual(r2.eigenvalues[i]);
      }
    });

    it('does not leak tensors', () => {
      const before = tf.memory().numTensors;

      const matrix = tf.tensor2d([
        [4, 1, 0],
        [1, 3, 1],
        [0, 1, 2],
      ]);
      const result = lanczos_smallest_eigenpairs(matrix, 2, { random_seed: 42, is_psd: false });
      matrix.dispose();

      const after = tf.memory().numTensors;

      // Lanczos operates on JS arrays, so no tensor leaks expected
      expect(after).toBe(before);
      // Verify result is valid
      expect(result.eigenvalues).toHaveLength(2);
    });

    it('throws for non-square matrix', () => {
      const rect = tf.tensor2d([[1, 2, 3], [4, 5, 6]]);
      expect(() => lanczos_smallest_eigenpairs(rect, 1)).toThrow('square');
      rect.dispose();
    });

    it('clamps negative eigenvalues to 0 when is_psd=true', () => {
      // [[0,1],[1,0]] is symmetric but indefinite: eigenvalues -1 and +1
      const matrix = tf.tensor2d([[0, 1], [1, 0]]);
      const result = lanczos_smallest_eigenpairs(matrix, 1, {
        random_seed: 42,
        is_psd: true,
      });
      matrix.dispose();
      expect(result.eigenvalues[0]).toBe(0);
    });

    it('preserves negative eigenvalues when is_psd=false', () => {
      // [[0,1],[1,0]] has eigenvalue -1 which must not be clamped
      const matrix = tf.tensor2d([[0, 1], [1, 0]]);
      const result = lanczos_smallest_eigenpairs(matrix, 1, {
        random_seed: 42,
        is_psd: false,
      });
      matrix.dispose();
      expect(result.eigenvalues[0]).toBeCloseTo(-1, 4);
    });

    it('returns orthonormal eigenvectors', () => {
      const n = 4;
      const k = 3;
      const matrix = tf.tensor2d([
        [4, 1, 0, 0],
        [1, 3, 1, 0],
        [0, 1, 2, 1],
        [0, 0, 1, 1],
      ]);
      const { eigenvectors } = lanczos_smallest_eigenpairs(matrix, k, {
        random_seed: 42,
        is_psd: false,
      });
      matrix.dispose();

      for (let a = 0; a < k; a++) {
        let self_dot = 0;
        for (let row = 0; row < n; row++) self_dot += eigenvectors[row][a] ** 2;
        expect(self_dot).toBeCloseTo(1, 4);
        for (let b = a + 1; b < k; b++) {
          let cross = 0;
          for (let row = 0; row < n; row++) cross += eigenvectors[row][a] * eigenvectors[row][b];
          expect(Math.abs(cross)).toBeLessThan(1e-4);
        }
      }
    });
  });
});
