const { SpectralClustering } = require('./dist/index.js');
const fs = require('fs');
const path = require('path');

// ARI calculation
function adjustedRandIndex(labelsA, labelsB) {
  if (labelsA.length !== labelsB.length) {
    throw new Error('Label arrays must have same length');
  }

  const n = labelsA.length;
  const labelToIndexA = new Map();
  const labelToIndexB = new Map();

  let nextA = 0;
  let nextB = 0;

  const contingency = [];

  for (let i = 0; i < n; i++) {
    const a = labelsA[i];
    const b = labelsB[i];

    if (!labelToIndexA.has(a)) {
      labelToIndexA.set(a, nextA++);
      contingency.push([]);
    }
    const idxA = labelToIndexA.get(a);

    if (!labelToIndexB.has(b)) {
      labelToIndexB.set(b, nextB++);
      for (const row of contingency) {
        while (row.length < nextB) row.push(0);
      }
    }
    const idxB = labelToIndexB.get(b);

    while (contingency[idxA].length <= idxB) {
      contingency[idxA].push(0);
    }

    contingency[idxA][idxB] = (contingency[idxA][idxB] || 0) + 1;
  }

  const ai = contingency.map((row) => row.reduce((s, v) => s + v, 0));
  const bj = contingency[0].map((_, j) => contingency.reduce((s, row) => s + row[j], 0));

  const comb2 = (x) => (x * (x - 1)) / 2;

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

async function testFixtures() {
  const FIXTURE_DIR = path.join(__dirname, 'test/fixtures/spectral');
  const files = fs.readdirSync(FIXTURE_DIR).filter(f => f.endsWith('.json'));
  
  console.log('Testing SpectralClustering fixtures:\n');
  
  for (const file of files) {
    const fixture = JSON.parse(fs.readFileSync(path.join(FIXTURE_DIR, file), 'utf-8'));
    
    const ctorParams = {
      nClusters: fixture.params.nClusters,
      affinity: fixture.params.affinity,
      randomState: fixture.params.randomState,
    };
    if (fixture.params.gamma !== undefined && fixture.params.gamma !== null) {
      ctorParams.gamma = fixture.params.gamma;
    }
    if (fixture.params.nNeighbors !== undefined && fixture.params.nNeighbors !== null) {
      ctorParams.nNeighbors = fixture.params.nNeighbors;
    }

    const model = new SpectralClustering(ctorParams);
    const ours = await model.fitPredict(fixture.X);
    const ari = adjustedRandIndex(ours, fixture.labels);
    
    console.log(`${file.padEnd(25)} ARI = ${ari.toFixed(4)} ${ari >= 0.95 ? '✓' : '✗'}`);
  }
}

testFixtures().catch(console.error);