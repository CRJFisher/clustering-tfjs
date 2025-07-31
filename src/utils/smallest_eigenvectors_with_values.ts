import * as tf from "@tensorflow/tfjs-node";
import { deterministic_eigenpair_processing } from "./eigen_post";

/**
 * Returns the `k` smallest eigenvectors AND eigenvalues of the provided symmetric matrix.
 * This is needed to apply diffusion map scaling like sklearn does.
 */
export function smallest_eigenvectors_with_values(
  matrix: tf.Tensor2D,
  k: number,
): { eigenvectors: tf.Tensor2D; eigenvalues: tf.Tensor1D } {
  if (!Number.isInteger(k) || k < 1) {
    throw new Error("k must be a positive integer.");
  }

  return tf.tidy(() => {
    // Import improved solver for better accuracy
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { improved_jacobi_eigen } = require("./eigen_improved");
    
    // 1) Full eigendecomposition with improved solver
    // For normalized Laplacians, we know it's PSD
    const { eigenvalues, eigenvectors } = improved_jacobi_eigen(matrix, {
      isPSD: true,
      maxIterations: 3000,
      tolerance: 1e-14,
    });

    // 2) Deterministic ordering & sign fixing
    const processed = deterministic_eigenpair_processing({
      eigenvalues,
      eigenvectors,
    });

    // 3) Determine number of numerically-zero eigenvalues
    const TOL = 1e-2;
    let c = 0;
    for (const v of processed.eigenvalues) {
      if (v <= TOL) c += 1;
      else break;
    }

    const n = processed.eigenvectors.length;
    const sliceCols = Math.min(k + c, n);
    
    // Extract selected eigenvectors
    const selectedVecs: number[][] = Array.from({ length: n }, () =>
      new Array(sliceCols),
    );
    const selectedVals: number[] = new Array(sliceCols);

    for (let col = 0; col < sliceCols; col++) {
      selectedVals[col] = processed.eigenvalues[col];
      for (let row = 0; row < n; row++) {
        selectedVecs[row][col] = processed.eigenvectors[row][col];
      }
    }

    return {
      eigenvectors: tf.tensor2d(selectedVecs, [n, sliceCols], "float32"),
      eigenvalues: tf.tensor1d(selectedVals, "float32"),
    };
  });
}