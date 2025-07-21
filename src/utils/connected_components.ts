import * as tf from "@tensorflow/tfjs-node";

/**
 * Detects the number of connected components in a graph based on its affinity matrix.
 * 
 * A graph is considered fully connected if there's only 1 component.
 * Multiple components are detected by counting near-zero eigenvalues of the Laplacian.
 * 
 * @param affinity - Affinity/adjacency matrix (n x n)
 * @param tolerance - Tolerance for detecting zero eigenvalues (default: 1e-2)
 * @returns Object containing:
 *   - numComponents: Number of connected components
 *   - isFullyConnected: Whether the graph has only 1 component
 */
export function detectConnectedComponents(
  affinity: tf.Tensor2D,
  tolerance: number = 1e-2
): { numComponents: number; isFullyConnected: boolean } {
  return tf.tidy(() => {
    // Import Laplacian computation
    const { normalised_laplacian } = require("./laplacian");
    
    // Compute normalized Laplacian
    const laplacian = normalised_laplacian(affinity);
    
    // Get eigenvalues using our improved eigen solver
    // We only need a few smallest eigenvalues to count components
    const k = Math.min(10, laplacian.shape[0]); // Check at most 10 smallest eigenvalues
    const { improved_jacobi_eigen } = require("./eigen_improved");
    
    const { eigenvalues } = improved_jacobi_eigen(laplacian, {
      isPSD: true,
      maxIterations: 1000,
      tolerance: 1e-10,
    });
    
    // Count eigenvalues that are approximately zero
    let numComponents = 0;
    
    // eigenvalues is already sorted in ascending order
    for (let i = 0; i < Math.min(k, eigenvalues.length); i++) {
      if (Math.abs(eigenvalues[i]) < tolerance) {
        numComponents++;
      } else {
        break; // Stop counting once we hit non-zero eigenvalues
      }
    }
    
    // If no zero eigenvalues found, assume 1 component (fully connected)
    if (numComponents === 0) {
      numComponents = 1;
    }
    
    return {
      numComponents,
      isFullyConnected: numComponents === 1
    };
  });
}

/**
 * Issues a warning if the graph is not fully connected, similar to sklearn.
 * 
 * @param affinity - Affinity matrix to check
 * @param tolerance - Tolerance for detecting zero eigenvalues
 * @returns true if graph is fully connected, false otherwise
 */
export function checkGraphConnectivity(
  affinity: tf.Tensor2D,
  tolerance: number = 1e-2
): boolean {
  const { isFullyConnected } = detectConnectedComponents(affinity, tolerance);
  
  if (!isFullyConnected) {
    console.warn(
      "Graph is not fully connected, spectral embedding may not work as expected."
    );
  }
  
  return isFullyConnected;
}