import fs from 'fs';
import path from 'path';
import { SpectralClustering } from '../src';

// Adjusted Rand Index function (same as tests)
function adjustedRandIndex(labelsA: number[], labelsB: number[]): number {
  if (labelsA.length !== labelsB.length) {
    throw new Error('Label arrays must have same length');
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
      for (const row of contingency) row.push(0);
    }
    const idxB = labelToIndexB.get(b)!;

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

async function run() {
  const dir = path.join(__dirname, '../test/fixtures/spectral');
  const files = fs.readdirSync(dir).filter((f) => f.endsWith('.json'));

  for (const file of files) {
    const fixture = JSON.parse(fs.readFileSync(path.join(dir, file), 'utf8')) as any;

    const ctor: any = {
      nClusters: fixture.params.nClusters,
      affinity: fixture.params.affinity,
      randomState: fixture.params.randomState,
    };
    if (fixture.params.gamma != null) ctor.gamma = fixture.params.gamma;
    if (fixture.params.nNeighbors != null) ctor.nNeighbors = fixture.params.nNeighbors;

    const model = new SpectralClustering(ctor);
    const ours = (await model.fitPredict(fixture.X)) as number[];
    const ari = adjustedRandIndex(ours, fixture.labels);
    console.log(`${file}: ARI = ${ari.toFixed(3)} | unique ${Array.from(new Set(ours)).length}`);
  }
}

run();

