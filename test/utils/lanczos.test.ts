import * as tf from '../../src/tf-adapter';
import { lanczos_smallest_eigenpairs } from '../../src/utils/lanczos';
import { improved_jacobi_eigen } from '../../src/utils/eigen_improved';
import { normalisedLaplacian } from '../../src/utils/laplacian';
import { make_random_stream } from '../../src/utils/rng/index';

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

      const result = lanczos_smallest_eigenpairs(matrix, 3, { randomSeed: 42 });
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

      const result = lanczos_smallest_eigenpairs(matrix, 1, { randomSeed: 42, isPSD: false });
      matrix.dispose();

      expect(result.eigenvalues).toHaveLength(1);
      expect(result.eigenvalues[0]).toBeCloseTo(1, 5);
    });

    it('finds all eigenvalues when k equals n', () => {
      const matrix = tf.tensor2d([
        [4, 1, 0],
        [1, 3, 1],
        [0, 1, 2],
      ]);

      const result = lanczos_smallest_eigenpairs(matrix, 3, { randomSeed: 42, isPSD: false });
      matrix.dispose();

      // Compare with Jacobi
      const matrixAgain = tf.tensor2d([
        [4, 1, 0],
        [1, 3, 1],
        [0, 1, 2],
      ]);
      const jacobi = improved_jacobi_eigen(matrixAgain);
      matrixAgain.dispose();

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

      const L = normalisedLaplacian(A) as tf.Tensor2D;
      A.dispose();

      const result = lanczos_smallest_eigenpairs(L, 2, {
        randomSeed: 42,
        isPSD: true,
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

      const L = normalisedLaplacian(A) as tf.Tensor2D;
      A.dispose();

      const result = lanczos_smallest_eigenpairs(L, 3, {
        randomSeed: 42,
        isPSD: true,
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

      const lanczosResult = lanczos_smallest_eigenpairs(matrix, k, {
        randomSeed: 42,
        isPSD: true,
      });

      const jacobiResult = improved_jacobi_eigen(matrix, {
        isPSD: true,
        maxIterations: 3000,
        tolerance: 1e-14,
      });
      matrix.dispose();

      // Compare eigenvalues (Jacobi returns all n, sorted ascending)
      for (let i = 0; i < k; i++) {
        expect(lanczosResult.eigenvalues[i]).toBeCloseTo(
          jacobiResult.eigenvalues[i],
          2,
        );
      }

      // Compare eigenvectors via |dot product| ≈ 1 for non-degenerate eigenvalues
      for (let col = 0; col < k; col++) {
        let dot = 0;
        for (let row = 0; row < n; row++) {
          dot +=
            lanczosResult.eigenvectors[row][col] *
            jacobiResult.eigenvectors[row][col];
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
      const result = lanczos_smallest_eigenpairs(matrix, 3, { randomSeed: 42 });
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

      const r1 = lanczos_smallest_eigenpairs(matrix, 2, { randomSeed: 42, isPSD: false });
      const r2 = lanczos_smallest_eigenpairs(matrix, 2, { randomSeed: 42, isPSD: false });
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
      const result = lanczos_smallest_eigenpairs(matrix, 2, { randomSeed: 42, isPSD: false });
      matrix.dispose();

      const after = tf.memory().numTensors;

      // Lanczos operates on JS arrays, so no tensor leaks expected
      expect(after).toBe(before);
      // Verify result is valid
      expect(result.eigenvalues).toHaveLength(2);
    });
  });
});
