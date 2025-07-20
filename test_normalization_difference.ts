import * as tf from '@tensorflow/tfjs-node';

async function testNormalizationDifference() {
  // Create test vectors with different norms
  const vectors = tf.tensor2d([
    [1.0, 0.0],           // Normal vector
    [0.0, 0.0],           // Zero vector
    [1e-12, 0.0],         // Near-zero vector
    [0.1, 0.1],           // Small vector
  ]);
  
  const eps = 1e-10;
  
  console.log('Original vectors:');
  console.log(await vectors.array());
  
  // Method 1: Current implementation (norm + eps)
  const norm1 = vectors.norm('euclidean', 1).expandDims(1);
  const normalized1 = vectors.div(norm1.add(eps));
  
  console.log('\nMethod 1 - div by (norm + eps):');
  console.log('Norms + eps:', await norm1.add(eps).squeeze().array());
  console.log('Normalized vectors:');
  console.log(await normalized1.array());
  
  // Method 2: Sklearn style (max(norm, eps))
  const norm2 = vectors.norm('euclidean', 1).expandDims(1);
  const normalized2 = vectors.div(tf.maximum(norm2, eps));
  
  console.log('\nMethod 2 - div by max(norm, eps):');
  console.log('max(norm, eps):', await tf.maximum(norm2, eps).squeeze().array());
  console.log('Normalized vectors:');
  console.log(await normalized2.array());
  
  // Compare differences
  console.log('\nDifferences:');
  for (let i = 0; i < 4; i++) {
    const v1 = await normalized1.slice([i, 0], [1, -1]).array();
    const v2 = await normalized2.slice([i, 0], [1, -1]).array();
    const diff = Math.sqrt((v1[0][0] - v2[0][0])**2 + (v1[0][1] - v2[0][1])**2);
    console.log(`Vector ${i}: difference = ${diff.toExponential(3)}`);
  }
  
  // Key insight: For zero/near-zero vectors
  console.log('\nKey insight for zero vector:');
  const zeroNorm = await norm1.slice([1, 0], [1, 1]).array();
  console.log(`Zero vector norm: ${zeroNorm[0][0]}`);
  console.log(`Method 1 divisor: ${zeroNorm[0][0] + eps} (changes direction!)`);
  console.log(`Method 2 divisor: ${Math.max(zeroNorm[0][0], eps)} (preserves zero)`);
  
  vectors.dispose();
  norm1.dispose();
  norm2.dispose();
  normalized1.dispose();
  normalized2.dispose();
}

testNormalizationDifference().catch(console.error);