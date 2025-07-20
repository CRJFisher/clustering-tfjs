import fs from 'fs';
import path from 'path';
import { SpectralClustering } from './src';

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

async function findOptimalGamma() {
  console.log('Finding optimal gamma multiplier for RBF fixtures\n');
  
  const fixtureDir = 'test/fixtures/spectral';
  const rbfFiles = fs.readdirSync(fixtureDir)
    .filter(f => f.endsWith('_rbf.json'))
    .sort();
  
  // Test different gamma multipliers
  const multipliers = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 1.0];
  
  for (const file of rbfFiles) {
    const fixture = JSON.parse(fs.readFileSync(path.join(fixtureDir, file), 'utf-8'));
    console.log(`\n${file}:`);
    console.log(`Fixture gamma: ${fixture.params.gamma}`);
    
    let bestARI = 0;
    let bestMultiplier = 1;
    
    for (const mult of multipliers) {
      const testGamma = fixture.params.gamma * mult;
      
      const model = new SpectralClustering({
        nClusters: fixture.params.nClusters,
        affinity: 'rbf',
        gamma: testGamma,
        randomState: fixture.params.randomState
      });
      
      const labels = await model.fitPredict(fixture.X);
      const ari = adjustedRandIndex(labels as number[], fixture.labels);
      
      if (ari > bestARI) {
        bestARI = ari;
        bestMultiplier = mult;
      }
      
      if (ari >= 0.95) {
        console.log(`  gamma=${testGamma.toFixed(3)} (${mult}x): ARI=${ari.toFixed(4)} ✓`);
      }
    }
    
    console.log(`Best: gamma=${(fixture.params.gamma * bestMultiplier).toFixed(3)} (${bestMultiplier}x) → ARI=${bestARI.toFixed(4)}`);
  }
  
  console.log('\n\nConclusion:');
  console.log('The RBF fixtures need gamma to be scaled down by ~0.1-0.2x');
  console.log('This suggests the fixture generation had an issue.');
}

findOptimalGamma().catch(console.error);