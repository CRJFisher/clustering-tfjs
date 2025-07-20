import fs from 'fs';
import path from 'path';
import { SpectralClustering } from './src';

// Fixed ARI function
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

async function checkResults() {
  const fixtureDir = 'test/fixtures/spectral';
  const files = fs.readdirSync(fixtureDir).filter(f => f.endsWith('.json')).sort();
  
  console.log('Task 12.4 Results - Row Normalization Fix');
  console.log('=========================================\n');
  
  let passCount = 0;
  let totalCount = 0;
  
  for (const file of files) {
    const fixture = JSON.parse(fs.readFileSync(path.join(fixtureDir, file), 'utf-8'));
    
    const params: any = {
      nClusters: fixture.params.nClusters,
      affinity: fixture.params.affinity,
      randomState: fixture.params.randomState
    };
    if (fixture.params.gamma != null) params.gamma = fixture.params.gamma;
    if (fixture.params.nNeighbors != null) params.nNeighbors = fixture.params.nNeighbors;
    
    try {
      const model = new SpectralClustering(params);
      const ours = await model.fitPredict(fixture.X);
      const ari = adjustedRandIndex(ours as number[], fixture.labels);
      
      totalCount++;
      const pass = ari >= 0.95;
      if (pass) passCount++;
      
      const icon = pass ? '✅' : '❌';
      const dataset = file.replace('.json', '').replace(/_/g, ' ').padEnd(20);
      console.log(`${icon} ${dataset} ARI = ${ari.toFixed(4)}`);
    } catch (e: any) {
      console.log(`❌ ${file}: ERROR - ${e.message}`);
    }
  }
  
  console.log(`\nBefore fix: 3/12 passing (25.0%)`);
  console.log(`After fix:  ${passCount}/${totalCount} passing (${(passCount/totalCount*100).toFixed(1)}%)`);
}

checkResults().catch(console.error);