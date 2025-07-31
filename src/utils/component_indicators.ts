import * as tf from "./tensorflow";

/**
 * Creates component indicator features for disconnected graphs.
 * 
 * For a graph with k connected components, this creates indicator features
 * where each feature has a constant value for all nodes in that component.
 * This mimics the behavior of sklearn's shift-invert eigenvectors.
 * 
 * @param componentLabels - Array indicating which component each node belongs to
 * @param numComponents - Total number of components detected
 * @param maxIndicators - Maximum number of indicator vectors to create (usually nClusters)
 * @returns Component indicator matrix (n_samples x min(numComponents, maxIndicators))
 */
export function createComponentIndicators(
  componentLabels: Int32Array,
  numComponents: number,
  maxIndicators: number
): tf.Tensor2D {
  return tf.tidy(() => {
    const n = componentLabels.length;
    const numIndicators = Math.min(numComponents, maxIndicators);
    
    // Count nodes per component for normalization
    const componentSizes = new Array(numComponents).fill(0);
    for (let i = 0; i < n; i++) {
      componentSizes[componentLabels[i]]++;
    }
    
    // Create indicator matrix
    const indicators = new Float32Array(n * numIndicators);
    
    // Fill indicators with normalized values
    // Using 1/sqrt(component_size) normalization to match eigenvector normalization
    for (let i = 0; i < n; i++) {
      const comp = componentLabels[i];
      if (comp < numIndicators) {
        indicators[i * numIndicators + comp] = 1.0 / Math.sqrt(componentSizes[comp]);
      }
    }
    
    return tf.tensor2d(indicators, [n, numIndicators], "float32");
  });
}