import * as tf from "../../test_support/tensorflow_helper";

describe("Row normalization methods", () => {
  const eps = 1e-10;
  
  it("should preserve direction with max(norm, eps)", () => {
    // Test with various vectors including zero and near-zero
    const vectors = tf.tensor2d([
      [1.0, 0.0],          // Normal vector
      [0.0, 0.0],          // Zero vector
      [1e-12, 0.0],        // Near-zero vector
      [0.1, 0.1],          // Small vector
    ]);
    
    // Method 1: Old way (norm + eps)
    const norm1 = vectors.norm("euclidean", 1).expandDims(1);
    const old_way = vectors.div(norm1.add(eps));
    const old_result = old_way.arraySync() as number[][];
    
    // Method 2: New way (max(norm, eps))
    const norm2 = vectors.norm("euclidean", 1).expandDims(1);
    const new_way = vectors.div(tf.maximum(norm2, eps));
    const new_result = new_way.arraySync() as number[][];
    
    // Zero vector should remain zero
    expect(new_result[1][0]).toBe(0);
    expect(new_result[1][1]).toBe(0);
    
    // Near-zero vector should have larger magnitude with new method
    const near_zero_mag_old = Math.sqrt(old_result[2][0]**2 + old_result[2][1]**2);
    const near_zero_mag_new = Math.sqrt(new_result[2][0]**2 + new_result[2][1]**2);
    expect(near_zero_mag_new).toBeGreaterThan(near_zero_mag_old);
    
    // Normal vectors should be unchanged
    expect(new_result[0][0]).toBeCloseTo(old_result[0][0], 10);
    expect(new_result[3][0]).toBeCloseTo(old_result[3][0], 10);
    
    vectors.dispose();
    norm1.dispose();
    norm2.dispose();
    old_way.dispose();
    new_way.dispose();
  });
  
  it("should handle edge cases correctly", () => {
    // Test edge cases
    const edge_cases = tf.tensor2d([
      [eps/2, 0],          // Smaller than eps
      [eps, 0],            // Exactly eps
      [eps*2, 0],          // Slightly larger than eps
    ]);
    
    const norms = edge_cases.norm("euclidean", 1).expandDims(1);
    const normalized = edge_cases.div(tf.maximum(norms, eps));
    const result = normalized.arraySync() as number[][];
    
    // Check normalization results
    // For vectors smaller than eps, they get scaled up significantly
    const mag0 = Math.sqrt(result[0][0]**2 + result[0][1]**2);
    const mag1 = Math.sqrt(result[1][0]**2 + result[1][1]**2);
    const mag2 = Math.sqrt(result[2][0]**2 + result[2][1]**2);
    
    // When norm < eps, we divide by eps, so magnitude = norm/eps
    expect(mag0).toBeCloseTo(0.5, 5);  // (eps/2) / eps = 0.5
    expect(mag1).toBeCloseTo(1.0, 5);  // eps / eps = 1.0
    expect(mag2).toBeCloseTo(1.0, 5);  // (eps*2) normalized to unit length
    
    edge_cases.dispose();
    norms.dispose();
    normalized.dispose();
  });
});