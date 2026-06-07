import * as tf from "../../test_support/tensorflow_helper";
import { compute_knn_affinity } from "./affinity";

describe("k-NN affinity connectivity", () => {
  it("should include self-loops by default", async () => {
    // Simple 4-point dataset
    const points = tf.tensor2d([
      [0, 0],
      [1, 0],
      [10, 0],  // Far from others
      [11, 0],  // Far from others
    ]);
    
    const { affinity } = compute_knn_affinity(points, 1, true);
    const affinity_array = await affinity.array();
    
    // Check diagonal has self-loops
    expect(affinity_array[0][0]).toBe(1);
    expect(affinity_array[1][1]).toBe(1);
    expect(affinity_array[2][2]).toBe(1);
    expect(affinity_array[3][3]).toBe(1);
    
    affinity.dispose();
    points.dispose();
  });
  
  it("should handle disconnected components with self-loops", async () => {
    // Two clearly separated clusters
    const points = tf.tensor2d([
      // Cluster 1
      [0, 0],
      [0.1, 0],
      [0, 0.1],
      // Cluster 2 (far away)
      [100, 0],
      [100.1, 0],
      [100, 0.1],
    ]);
    
    const { affinity } = compute_knn_affinity(points, 2, true);
    const affinity_array = await affinity.array();
    
    // Each point should connect to itself
    for (let i = 0; i < 6; i++) {
      expect(affinity_array[i][i]).toBe(1);
    }
    
    // Check no connections between clusters
    // Points 0-2 should not connect to points 3-5
    for (let i = 0; i < 3; i++) {
      for (let j = 3; j < 6; j++) {
        expect(affinity_array[i][j]).toBe(0);
        expect(affinity_array[j][i]).toBe(0);
      }
    }
    
    affinity.dispose();
    points.dispose();
  });
  
  it("should exclude self-loops when includeSelf=false", async () => {
    const points = tf.tensor2d([
      [0, 0],
      [1, 0],
      [2, 0],
    ]);
    
    const { affinity } = compute_knn_affinity(points, 1, false);
    const affinity_array = await affinity.array();
    
    // Check diagonal has no self-loops
    expect(affinity_array[0][0]).toBe(0);
    expect(affinity_array[1][1]).toBe(0);
    expect(affinity_array[2][2]).toBe(0);
    
    // But should still have connections to neighbors
    expect(affinity_array[0][1]).toBe(1); // 0 -> 1
    expect(affinity_array[1][0]).toBe(1); // 1 -> 0 (symmetric)
    
    affinity.dispose();
    points.dispose();
  });
});