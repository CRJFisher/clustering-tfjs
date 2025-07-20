import fs from 'fs';
import { SpectralClustering } from '../src';

function adjustedRandIndex(labelsA: number[], labelsB: number[]): number {
  if (labelsA.length !== labelsB.length) throw new Error('length mismatch');
  const n = labelsA.length;
  const mapA = new Map<number, number>();
  const mapB = new Map<number, number>();
  const cont: number[][] = [];
  let nextA = 0, nextB = 0;
  for (let i = 0; i < n; i++) {
    const a = labelsA[i];
    const b = labelsB[i];
    if (!mapA.has(a)) { mapA.set(a, nextA++); cont.push([]); }
    if (!mapB.has(b)) { mapB.set(b, nextB++); cont.forEach(row => row.push(0)); }
    const ia = mapA.get(a)!; const ib = mapB.get(b)!;
    cont[ia][ib] = (cont[ia][ib] || 0) + 1;
  }
  const comb2 = (x: number) => (x * (x - 1)) / 2;
  let sum = 0;
  for (const row of cont) for (const val of row) sum += comb2(val);
  const ai = cont.map(row => row.reduce((s, v) => s + v, 0));
  const bj = cont[0].map((_, j) => cont.reduce((s, row) => s + row[j], 0));
  const sumAi = ai.reduce((s, v) => s + comb2(v), 0);
  const sumBj = bj.reduce((s, v) => s + comb2(v), 0);
  const expected = (sumAi * sumBj) / comb2(n);
  const max = (sumAi + sumBj) / 2;
  if (max === expected) return 0;
  return (sum - expected) / (max - expected);
}

const file = process.argv[2];
if (!file) throw new Error('pass fixture file');
const fixture = JSON.parse(fs.readFileSync(file, 'utf-8'));
const ctor: any = {
  nClusters: fixture.params.nClusters,
  affinity: fixture.params.affinity,
  randomState: fixture.params.randomState,
};
if (fixture.params.gamma != null) ctor.gamma = fixture.params.gamma;
if (fixture.params.nNeighbors != null) ctor.nNeighbors = fixture.params.nNeighbors;

(async () => {
  const model = new SpectralClustering(ctor);
  const ours = (await model.fitPredict(fixture.X)) as number[];
  console.log('ours', ours.slice(0,50));
  console.log('ref', fixture.labels.slice(0,50));
  console.log('ari', adjustedRandIndex(ours, fixture.labels));
})();

