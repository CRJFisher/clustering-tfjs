import * as tf from '@tensorflow/tfjs-node';

async function debugLabelMismatch() {
  // Load the failing fixture
  const fixture = require('./test/fixtures/spectral/circles_n2_rbf.json');
  const { SpectralClustering } = require('./src');
  
  const params: any = {
    nClusters: fixture.params.nClusters,
    affinity: fixture.params.affinity,
    randomState: fixture.params.randomState
  };
  if (fixture.params.gamma != null) params.gamma = fixture.params.gamma;
  
  const model = new SpectralClustering(params);
  const ourLabels = await model.fitPredict(fixture.X);
  const expectedLabels = fixture.labels;
  
  console.log('Label comparison for circles_n2_rbf');
  console.log('===================================\n');
  
  // Find mismatches
  const mismatches: number[] = [];
  for (let i = 0; i < ourLabels.length; i++) {
    if (ourLabels[i] !== expectedLabels[i]) {
      mismatches.push(i);
    }
  }
  
  console.log(`Total samples: ${ourLabels.length}`);
  console.log(`Mismatches: ${mismatches.length} (${(mismatches.length/ourLabels.length*100).toFixed(1)}%)\n`);
  
  // Show first few mismatches
  console.log('First 10 mismatches:');
  console.log('Index | Ours | Expected');
  console.log('-'.repeat(25));
  for (let i = 0; i < Math.min(10, mismatches.length); i++) {
    const idx = mismatches[i];
    console.log(`${idx.toString().padStart(5)} | ${ourLabels[idx].toString().padStart(4)} | ${expectedLabels[idx].toString().padStart(8)}`);
  }
  
  // Check if it's just a label flip
  const flippedLabels = ourLabels.map(l => l === 0 ? 1 : 0);
  let flippedMismatches = 0;
  for (let i = 0; i < flippedLabels.length; i++) {
    if (flippedLabels[i] !== expectedLabels[i]) {
      flippedMismatches++;
    }
  }
  
  console.log(`\nMismatches with flipped labels: ${flippedMismatches}`);
  
  // Calculate ARI manually
  function adjustedRandIndex(labelsA: number[], labelsB: number[]): number {
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
    const bj = contingency[0].map((_, j) => contingency.reduce((s, row) => s + row[j], 0));

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
  
  const ari = adjustedRandIndex(ourLabels as number[], expectedLabels);
  const ariFlipped = adjustedRandIndex(flippedLabels as number[], expectedLabels);
  
  console.log(`\nARI with original labels: ${ari.toFixed(4)}`);
  console.log(`ARI with flipped labels: ${ariFlipped.toFixed(4)}`);
  
  // Print contingency table
  console.log('\nContingency table:');
  const cont = [[0, 0], [0, 0]];
  for (let i = 0; i < ourLabels.length; i++) {
    cont[ourLabels[i]][expectedLabels[i]]++;
  }
  console.log('       Expected');
  console.log('       0    1');
  console.log(`Ours 0 ${cont[0][0].toString().padStart(4)} ${cont[0][1].toString().padStart(4)}`);
  console.log(`     1 ${cont[1][0].toString().padStart(4)} ${cont[1][1].toString().padStart(4)}`);
}

debugLabelMismatch().catch(console.error);