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
 *   - componentLabels: Array indicating which component each node belongs to
 */
export function detectConnectedComponents(
  affinity: tf.Tensor2D,
  tolerance: number = 1e-2
): { numComponents: number; isFullyConnected: boolean; componentLabels: Int32Array } {
  const n = affinity.shape[0];
  const componentLabels = new Int32Array(n).fill(-1);
  
  // Get affinity data for graph traversal
  const affinityData = affinity.arraySync();
  
  // BFS to find connected components
  let currentComponent = 0;
  
  for (let startNode = 0; startNode < n; startNode++) {
    if (componentLabels[startNode] !== -1) continue; // Already assigned
    
    // BFS from this node
    const queue: number[] = [startNode];
    componentLabels[startNode] = currentComponent;
    
    while (queue.length > 0) {
      const node = queue.shift()!;
      
      // Check all neighbors
      for (let neighbor = 0; neighbor < n; neighbor++) {
        if (neighbor !== node && 
            componentLabels[neighbor] === -1 && 
            affinityData[node][neighbor] > tolerance) {
          componentLabels[neighbor] = currentComponent;
          queue.push(neighbor);
        }
      }
    }
    
    currentComponent++;
  }
  
  const numComponents = currentComponent;
  
  return {
    numComponents,
    isFullyConnected: numComponents === 1,
    componentLabels
  };
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