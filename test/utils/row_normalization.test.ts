import * as tf from "../tensorflow-helper";

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
    const oldWay = vectors.div(norm1.add(eps));
    const oldResult = oldWay.arraySync() as number[][];
    
    // Method 2: New way (max(norm, eps))
    const norm2 = vectors.norm("euclidean", 1).expandDims(1);
    const newWay = vectors.div(tf.maximum(norm2, eps));
    const newResult = newWay.arraySync() as number[][];
    
    // Zero vector should remain zero
    expect(newResult[1][0]).toBe(0);
    expect(newResult[1][1]).toBe(0);
    
    // Near-zero vector should have larger magnitude with new method
    const nearZeroMagOld = Math.sqrt(oldResult[2][0]**2 + oldResult[2][1]**2);
    const nearZeroMagNew = Math.sqrt(newResult[2][0]**2 + newResult[2][1]**2);
    expect(nearZeroMagNew).toBeGreaterThan(nearZeroMagOld);
    
    // Normal vectors should be unchanged
    expect(newResult[0][0]).toBeCloseTo(oldResult[0][0], 10);
    expect(newResult[3][0]).toBeCloseTo(oldResult[3][0], 10);
    
    vectors.dispose();
    norm1.dispose();
    norm2.dispose();
    oldWay.dispose();
    newWay.dispose();
  });
  
  it("should handle edge cases correctly", () => {
    // Test edge cases
    const edgeCases = tf.tensor2d([
      [eps/2, 0],          // Smaller than eps
      [eps, 0],            // Exactly eps
      [eps*2, 0],          // Slightly larger than eps
    ]);
    
    const norms = edgeCases.norm("euclidean", 1).expandDims(1);
    const normalized = edgeCases.div(tf.maximum(norms, eps));
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
    
    edgeCases.dispose();
    norms.dispose();
    normalized.dispose();
  });
});