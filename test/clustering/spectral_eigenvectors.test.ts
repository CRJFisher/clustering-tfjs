import * as tf from "@tensorflow/tfjs-node";
import { SpectralClustering } from "../../src";

describe("SpectralClustering trivial eigenvector handling", () => {
  it("should handle disconnected components correctly", async () => {
    // Create 3 completely disconnected components
    const X = [
      // Component 1
      [0, 0],
      [0.1, 0],
      [0, 0.1],
      // Component 2
      [10, 0],
      [10.1, 0],
      [10, 0.1],
      // Component 3
      [20, 0],
      [20.1, 0],
      [20, 0.1],
    ];
    
    const model = new SpectralClustering({
      nClusters: 3,
      affinity: 'nearest_neighbors',
      nNeighbors: 2,
      randomState: 42
    });
    
    const labels = await model.fitPredict(X) as number[];
    
    // Each component should get its own unique label
    const comp1Labels = new Set([labels[0], labels[1], labels[2]]);
    const comp2Labels = new Set([labels[3], labels[4], labels[5]]);
    const comp3Labels = new Set([labels[6], labels[7], labels[8]]);
    
    expect(comp1Labels.size).toBe(1);
    expect(comp2Labels.size).toBe(1);
    expect(comp3Labels.size).toBe(1);
    
    // All three components should have different labels
    const allLabels = new Set(labels);
    expect(allLabels.size).toBe(3);
  });
  
  it("should throw error when insufficient informative eigenvectors", async () => {
    // Create a graph with only one informative eigenvector
    // (all points in a line with same distances)
    const X = [
      [0, 0],
      [1, 0],
      [2, 0],
      [3, 0],
    ];
    
    const model = new SpectralClustering({
      nClusters: 3,  // Want 3 clusters but will only have 1 informative eigenvector
      affinity: 'rbf',
      gamma: 0.0001,  // Very small gamma makes affinity nearly uniform
      randomState: 42
    });
    
    await expect(model.fitPredict(X)).rejects.toThrow(/constant eigenvectors/);
  });
  
  it("should handle graphs with multiple constant eigenvectors", async () => {
    // Create a graph that will have multiple near-zero eigenvalues
    // This is a case where we have some connected components but not enough for nClusters
    const X = [
      // Two connected components
      [0, 0], [0.1, 0],     // Component 1
      [10, 0], [10.1, 0],   // Component 2
    ];
    
    const model = new SpectralClustering({
      nClusters: 2,
      affinity: 'nearest_neighbors',
      nNeighbors: 1,
      randomState: 42
    });
    
    const labels = await model.fitPredict(X) as number[];
    
    // Should correctly identify 2 clusters
    expect(new Set(labels).size).toBe(2);
    expect(labels[0]).toBe(labels[1]); // Same component
    expect(labels[2]).toBe(labels[3]); // Same component
    expect(labels[0]).not.toBe(labels[2]); // Different components
  });
});