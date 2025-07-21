import * as tf from "@tensorflow/tfjs-node";

/**
 * Creates component indicator features for disconnected graphs.
 * 
 * For a graph with k connected components, this creates k indicator features
 * where each feature has a constant value for all nodes in that component.
 * 
 * This is an alternative to using eigenvectors when the graph is disconnected.
 * 
 * @param affinity - Affinity matrix
 * @param numComponents - Number of connected components 
 * @returns Component indicator matrix (n_samples x numComponents)
 */
export function createComponentIndicators(
  affinity: tf.Tensor2D,
  numComponents: number
): tf.Tensor2D {
  return tf.tidy(() => {
    const n = affinity.shape[0];
    
    // Initialize component labels
    const componentLabels = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      componentLabels[i] = -1; // Unassigned
    }
    
    // Get affinity data
    const affinityData = affinity.arraySync();
    
    // Find connected components using BFS
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
          if (affinityData[node][neighbor] > 0 && componentLabels[neighbor] === -1) {
            componentLabels[neighbor] = currentComponent;
            queue.push(neighbor);
          }
        }
      }
      
      currentComponent++;
    }
    
    // Create indicator matrix
    const indicators = Array(n).fill(null).map(() => Array(numComponents).fill(0));
    
    for (let i = 0; i < n; i++) {
      const comp = componentLabels[i];
      if (comp >= 0 && comp < numComponents) {
        // Set a unique value for each component to match sklearn's behavior
        // Using 1/sqrt(component_size) to normalize
        indicators[i][comp] = 1.0;
      }
    }
    
    // Normalize each indicator by component size
    const componentSizes = new Array(numComponents).fill(0);
    for (let i = 0; i < n; i++) {
      const comp = componentLabels[i];
      if (comp >= 0 && comp < numComponents) {
        componentSizes[comp]++;
      }
    }
    
    for (let i = 0; i < n; i++) {
      const comp = componentLabels[i];
      if (comp >= 0 && comp < numComponents) {
        indicators[i][comp] = 1.0 / Math.sqrt(componentSizes[comp]);
      }
    }
    
    return tf.tensor2d(indicators, [n, numComponents], "float32");
  });
}