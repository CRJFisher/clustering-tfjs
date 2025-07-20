const fs = require('fs');

// Load both embeddings
const ourData = JSON.parse(fs.readFileSync('our_embedding.json', 'utf-8'));
const sklearnData = JSON.parse(fs.readFileSync('tools/sklearn_fixtures/sklearn_embedding.json', 'utf-8'));

console.log('Comparing spectral embeddings between our implementation and sklearn\n');

// Basic stats
console.log('Shape comparison:');
console.log(`Our shape: ${ourData.shape[0]} x ${ourData.shape[1]}`);
console.log(`Sklearn shape: ${sklearnData.shape[0]} x ${sklearnData.shape[1]}\n`);

// Compare dimensions
const ourEmbed = ourData.embedding;
const skEmbed = sklearnData.embedding;

// Compare value ranges
const ourFlat = ourEmbed.flat();
const skFlat = skEmbed.flat();

console.log('Value ranges:');
console.log(`Our range: [${Math.min(...ourFlat).toFixed(6)}, ${Math.max(...ourFlat).toFixed(6)}]`);
console.log(`Sklearn range: [${Math.min(...skFlat).toFixed(6)}, ${Math.max(...skFlat).toFixed(6)}]`);
console.log(`Our mean: ${(ourFlat.reduce((a, b) => a + b) / ourFlat.length).toFixed(6)}`);
console.log(`Sklearn mean: ${(skFlat.reduce((a, b) => a + b) / skFlat.length).toFixed(6)}\n`);

// Key observation: our values are much larger!
const scaleRatio = Math.max(...ourFlat) / Math.max(...skFlat);
console.log(`Scale difference: Our values are ~${scaleRatio.toFixed(1)}x larger\n`);

// Compare norms
console.log('Row norm comparison (first 5 rows):');
for (let i = 0; i < 5; i++) {
  const ourNorm = Math.sqrt(ourEmbed[i].reduce((sum, v) => sum + v*v, 0));
  const skNorm = Math.sqrt(skEmbed[i].reduce((sum, v) => sum + v*v, 0));
  console.log(`Row ${i}: our=${ourNorm.toFixed(6)}, sklearn=${skNorm.toFixed(6)}, ratio=${(ourNorm/skNorm).toFixed(3)}`);
}

// Check if sklearn normalizes rows
console.log('\nChecking if sklearn normalizes embedding rows:');
let skNormalized = true;
for (let i = 0; i < skEmbed.length; i++) {
  const norm = Math.sqrt(skEmbed[i].reduce((sum, v) => sum + v*v, 0));
  if (Math.abs(norm - 1.0) > 0.01) {
    skNormalized = false;
    break;
  }
}
console.log(`Sklearn rows normalized to unit length: ${skNormalized}`);

// Direct comparison of first few rows
console.log('\nDirect value comparison (first 3 rows):');
for (let i = 0; i < 3; i++) {
  console.log(`\nRow ${i}:`);
  console.log(`  Our:     [${ourEmbed[i].map(v => v.toFixed(6)).join(', ')}]`);
  console.log(`  Sklearn: [${skEmbed[i].map(v => v.toFixed(6)).join(', ')}]`);
}

// Key finding
console.log('\n=== KEY FINDING ===');
console.log('Our embedding values are ~3x larger than sklearn\'s.');
console.log('This suggests sklearn might be applying additional normalization.');
console.log('\nPossible causes:');
console.log('1. Different eigenvector normalization (we use unit norm, sklearn might use something else)');
console.log('2. Different Laplacian normalization');
console.log('3. Additional scaling in sklearn\'s spectral embedding');

// Check affinity matrix
if ('affinity_min' in sklearnData) {
  console.log('\nAffinity matrix comparison:');
  console.log(`Sklearn affinity range: [${sklearnData.affinity_min.toFixed(6)}, ${sklearnData.affinity_max.toFixed(6)}]`);
  console.log('(Our affinity should be similar for KNN with symmetric averaging)');
}