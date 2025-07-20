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

async function testGammaFix() {
  const fixture = require('./test/fixtures/spectral/blobs_n2_rbf.json');
  
  console.log('Testing gamma fix for blobs_n2_rbf\n');
  console.log('Fixture gamma:', fixture.params.gamma);
  console.log('Number of features:', fixture.X[0].length);
  console.log('Sklearn default gamma:', 1 / fixture.X[0].length);
  
  // Test with fixture gamma
  console.log('\nWith fixture gamma (1.0):');
  const model1 = new SpectralClustering({
    nClusters: fixture.params.nClusters,
    affinity: 'rbf',
    gamma: fixture.params.gamma,
    randomState: fixture.params.randomState
  });
  const labels1 = await model1.fitPredict(fixture.X);
  const ari1 = adjustedRandIndex(labels1 as number[], fixture.labels);
  console.log('ARI:', ari1.toFixed(4));
  
  // Test with sklearn default
  console.log('\nWith sklearn default gamma (0.5):');
  const model2 = new SpectralClustering({
    nClusters: fixture.params.nClusters,
    affinity: 'rbf',
    gamma: 0.5,
    randomState: fixture.params.randomState
  });
  const labels2 = await model2.fitPredict(fixture.X);
  const ari2 = adjustedRandIndex(labels2 as number[], fixture.labels);
  console.log('ARI:', ari2.toFixed(4));
  
  // Test with various gamma values
  console.log('\nTrying other gamma values:');
  for (const gamma of [0.1, 0.2, 0.3, 0.7, 1.5, 2.0]) {
    const model = new SpectralClustering({
      nClusters: fixture.params.nClusters,
      affinity: 'rbf',
      gamma: gamma,
      randomState: fixture.params.randomState
    });
    const labels = await model.fitPredict(fixture.X);
    const ari = adjustedRandIndex(labels as number[], fixture.labels);
    console.log(`gamma=${gamma}: ARI=${ari.toFixed(4)}`);
  }
  
  // The issue might be that sklearn used a different gamma when generating the labels
  console.log('\n\nConclusion:');
  console.log('The fixture specifies gamma=1.0, but the labels might have been');
  console.log('generated with a different gamma value, or there might be another');
  console.log('algorithmic difference we haven\'t identified yet.');
}

testGammaFix().catch(console.error);