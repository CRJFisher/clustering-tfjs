import * as tf from '@tensorflow/tfjs-node';

async function testTFJSEigen() {
  console.log('Testing TensorFlow.js for eigendecomposition capabilities\n');
  
  // Check if eig operation exists
  console.log('Available TF operations:', Object.keys(tf).filter(k => k.toLowerCase().includes('eig')).sort());
  
  // Test simple matrix
  const A = tf.tensor2d([
    [4, -1, -1],
    [-1, 3, -1],
    [-1, -1, 2]
  ]);
  
  try {
    // Try tf.linalg namespace
    console.log('\ntf.linalg methods:', Object.keys(tf.linalg).sort());
    
    // Check for QR decomposition (could be used for eigenvalues)
    if ('qr' in tf.linalg) {
      console.log('\nQR decomposition available!');
      const [q, r] = tf.linalg.qr(A);
      console.log('Q shape:', q.shape);
      console.log('R shape:', r.shape);
      q.dispose();
      r.dispose();
    }
    
    // Check for other useful operations
    console.log('\nOther potentially useful operations:');
    console.log('- tf.matMul:', typeof tf.matMul);
    console.log('- tf.transpose:', typeof tf.transpose);
    console.log('- tf.diag:', typeof tf.diag);
    console.log('- tf.norm:', typeof tf.norm);
    
  } catch (e: any) {
    console.error('Error:', e.message);
  }
  
  A.dispose();
}

testTFJSEigen().catch(console.error);