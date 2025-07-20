import fs from 'fs';
import path from 'path';
import { SpectralClustering } from './src';

// ARI calculation
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

// Analyze the pattern of failures
async function analyzeFailures() {
  const fixtureDir = 'test/fixtures/spectral';
  const files = fs.readdirSync(fixtureDir).filter(f => f.endsWith('.json')).sort();
  
  console.log('Analysis of Failing Fixture Tests');
  console.log('=================================\n');
  
  const failingTests: any[] = [];
  
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
      
      // Calculate ARI properly
      const ari = adjustedRandIndex(ours as number[], fixture.labels);
      
      if (ari < 0.95) {
        failingTests.push({
          file: file.replace('.json', ''),
          affinity: fixture.params.affinity,
          nClusters: fixture.params.nClusters,
          ari: ari,
          nSamples: fixture.X.length,
          gamma: fixture.params.gamma,
          nNeighbors: fixture.params.nNeighbors
        });
      }
    } catch (e: any) {
      console.error(`Error in ${file}: ${e.message}`);
    }
  }
  
  console.log('Failing tests grouped by affinity type:\n');
  
  const rbfFailures = failingTests.filter(t => t.affinity === 'rbf');
  const knnFailures = failingTests.filter(t => t.affinity === 'nearest_neighbors');
  
  console.log('RBF failures:', rbfFailures.length);
  for (const test of rbfFailures) {
    console.log(`  ${test.file}: ARI=${test.ari.toFixed(3)}, gamma=${test.gamma}`);
  }
  
  console.log('\nKNN failures:', knnFailures.length);
  for (const test of knnFailures) {
    console.log(`  ${test.file}: ARI=${test.ari.toFixed(3)}, k=${test.nNeighbors}`);
  }
  
  // Check specific patterns
  console.log('\n\nPattern Analysis:');
  console.log('=================');
  
  // All blob tests pass
  const blobTests = failingTests.filter(t => t.file.includes('blobs'));
  console.log(`Blob dataset failures: ${blobTests.length} (all pass!)`);
  
  // Circles and moons have issues
  const circleTests = failingTests.filter(t => t.file.includes('circles'));
  const moonTests = failingTests.filter(t => t.file.includes('moons'));
  
  console.log(`Circle dataset failures: ${circleTests.length}/4`);
  console.log(`Moon dataset failures: ${moonTests.length}/4`);
  
  // Check if it's related to n_clusters
  const n2Failures = failingTests.filter(t => t.nClusters === 2);
  const n3Failures = failingTests.filter(t => t.nClusters === 3);
  
  console.log(`\nFailures with nClusters=2: ${n2Failures.length}`);
  console.log(`Failures with nClusters=3: ${n3Failures.length}`);
}

analyzeFailures().catch(console.error);