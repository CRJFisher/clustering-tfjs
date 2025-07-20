import fs from 'fs';

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

// Check if dimensions match
if (ourEmbed.length !== skEmbed.length || ourEmbed[0].length !== skEmbed[0].length) {
  console.error('ERROR: Embedding dimensions do not match!');
  process.exit(1);
}

// Compare value ranges
const ourFlat = ourEmbed.flat();
const skFlat = skEmbed.flat();

console.log('Value ranges:');
console.log(`Our range: [${Math.min(...ourFlat).toFixed(6)}, ${Math.max(...ourFlat).toFixed(6)}]`);
console.log(`Sklearn range: [${Math.min(...skFlat).toFixed(6)}, ${Math.max(...skFlat).toFixed(6)}]`);
console.log(`Our mean: ${(ourFlat.reduce((a, b) => a + b) / ourFlat.length).toFixed(6)}`);
console.log(`Sklearn mean: ${(skFlat.reduce((a, b) => a + b) / skFlat.length).toFixed(6)}\n`);

// Compare norms
console.log('Row norm comparison (first 5 rows):');
for (let i = 0; i < 5; i++) {
  const ourNorm = Math.sqrt(ourEmbed[i].reduce((sum, v) => sum + v*v, 0));
  const skNorm = Math.sqrt(skEmbed[i].reduce((sum, v) => sum + v*v, 0));
  console.log(`Row ${i}: our=${ourNorm.toFixed(6)}, sklearn=${skNorm.toFixed(6)}, ratio=${(ourNorm/skNorm).toFixed(3)}`);
}

// Check scaling difference
console.log('\nChecking if embeddings differ by a constant scale factor:');
const scaleFactors: number[] = [];
for (let i = 0; i < 5; i++) {
  for (let j = 0; j < ourEmbed[i].length; j++) {
    if (Math.abs(skEmbed[i][j]) > 1e-10) {
      scaleFactors.push(ourEmbed[i][j] / skEmbed[i][j]);
    }
  }
}

const avgScale = scaleFactors.reduce((a, b) => a + b) / scaleFactors.length;
const scaleStd = Math.sqrt(scaleFactors.reduce((sum, s) => sum + (s - avgScale)**2, 0) / scaleFactors.length);
console.log(`Average scale factor: ${avgScale.toFixed(6)} ± ${scaleStd.toFixed(6)}`);

// Check if eigenvectors might be flipped or reordered
console.log('\nChecking eigenvector alignment:');
for (let col = 0; col < ourEmbed[0].length; col++) {
  const ourCol = ourEmbed.map(row => row[col]);
  const skCol = skEmbed.map(row => row[col]);
  
  // Compute correlation
  const ourMean = ourCol.reduce((a, b) => a + b) / ourCol.length;
  const skMean = skCol.reduce((a, b) => a + b) / skCol.length;
  
  let cov = 0, ourVar = 0, skVar = 0;
  for (let i = 0; i < ourCol.length; i++) {
    cov += (ourCol[i] - ourMean) * (skCol[i] - skMean);
    ourVar += (ourCol[i] - ourMean) ** 2;
    skVar += (skCol[i] - skMean) ** 2;
  }
  
  const correlation = cov / Math.sqrt(ourVar * skVar);
  console.log(`Column ${col} correlation: ${correlation.toFixed(6)}`);
}

// Direct comparison of first few rows
console.log('\nDirect value comparison (first 3 rows):');
for (let i = 0; i < 3; i++) {
  console.log(`\nRow ${i}:`);
  console.log(`  Our:     [${ourEmbed[i].map(v => v.toFixed(6)).join(', ')}]`);
  console.log(`  Sklearn: [${skEmbed[i].map(v => v.toFixed(6)).join(', ')}]`);
  console.log(`  Diff:    [${ourEmbed[i].map((v, j) => (v - skEmbed[i][j]).toFixed(6)).join(', ')}]`);
}

// Check affinity matrix values (if available)
if ('affinity_min' in sklearnData) {
  console.log('\nAffinity matrix comparison:');
  console.log(`Sklearn affinity range: [${sklearnData.affinity_min.toFixed(6)}, ${sklearnData.affinity_max.toFixed(6)}]`);
}

// Compare final labels
console.log('\nLabel comparison:');
const ourLabels = ourData.labels;
const skLabels = sklearnData.labels;

// Compute ARI between our labels and sklearn labels
function adjustedRandIndex(labelsA: number[], labelsB: number[]): number {
  if (labelsA.length !== labelsB.length) {
    throw new Error("Label arrays must have same length");
  }

  const n = labelsA.length;
  const labelToIndexA = new Map<number, number>();
  const labelToIndexB = new Map<number, number>();

  let nextA = 0;
  let nextB = 0;

  const contingency: number[][] = [];

  for (let i = 0; i < n; i++) {
    const a = labelsA[i];
    const b = labelsB[i];

    if (!labelToIndexA.has(a)) {
      labelToIndexA.set(a, nextA++);
      contingency.push([]);
    }
    const idxA = labelToIndexA.get(a)!;

    if (!labelToIndexB.has(b)) {
      labelToIndexB.set(b, nextB++);
      for (const row of contingency) {
        while (row.length < nextB) row.push(0);
      }
    }
    const idxB = labelToIndexB.get(b)!;

    while (contingency[idxA].length <= idxB) {
      contingency[idxA].push(0);
    }
    
    contingency[idxA][idxB] = (contingency[idxA][idxB] || 0) + 1;
  }

  const ai = contingency.map((row) => row.reduce((s, v) => s + v, 0));
  const bj = contingency[0].map((_, j) => contingency.reduce((s, row) => s + (row[j] || 0), 0));

  const comb2 = (x: number): number => (x * (x - 1)) / 2;

  let sumComb = 0;
  for (const row of contingency) {
    for (const val of row) sumComb += comb2(val);
  }

  const sumAi = ai.reduce((s, v) => s + comb2(v), 0);
  const sumBj = bj.reduce((s, v) => s + comb2(v), 0);

  const expected = (sumAi * sumBj) / comb2(n);
  const max = (sumAi + sumBj) / 2;

  if (max === expected) return 0;
  return (sumComb - expected) / (max - expected);
}

const ari = adjustedRandIndex(ourLabels, skLabels);
console.log(`ARI between our labels and sklearn labels: ${ari.toFixed(6)}`);

// Summary
console.log('\n=== SUMMARY ===');
console.log('Key differences found:');
console.log(`1. Value scales differ by factor of ~${avgScale.toFixed(2)}`);
console.log(`2. Row norms: ours are ~${(ourFlat.reduce((a, b) => a + Math.abs(b), 0) / skFlat.reduce((a, b) => a + Math.abs(b), 0)).toFixed(2)}x larger`);
console.log(`3. Label agreement (ARI): ${ari.toFixed(3)}`);