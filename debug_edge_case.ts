import * as tf from '@tensorflow/tfjs-node';

const eps = 1e-10;

const edgeCases = tf.tensor2d([
  [eps/2, 0],          // Smaller than eps
  [eps, 0],            // Exactly eps
  [eps*2, 0],          // Slightly larger than eps
]);

console.log('Edge case vectors:');
console.log(edgeCases.arraySync());

const norms = edgeCases.norm("euclidean", 1);
console.log('\nNorms:', norms.arraySync());

const normsExpanded = norms.expandDims(1);
const maxNorms = tf.maximum(normsExpanded, eps);
console.log('\nmax(norm, eps):', maxNorms.squeeze().arraySync());

const normalized = edgeCases.div(maxNorms);
const result = normalized.arraySync();

console.log('\nNormalized:');
console.log(result);

console.log('\nMagnitudes:');
result.forEach((row: any, i: number) => {
  const mag = Math.sqrt(row[0]**2 + row[1]**2);
  console.log(`Vector ${i}: magnitude = ${mag}`);
});