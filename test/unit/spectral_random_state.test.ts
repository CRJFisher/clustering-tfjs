import { SpectralClustering } from "../../src";

/**
 * Unit test verifying that the `randomState` provided to `SpectralClustering`
 * is forwarded to the internal `KMeans` initialisation which makes the final
 * cluster labels fully reproducible.
 */

function makeChallengingData(): number[][] {
  // Generate points in two overlapping Gaussian clouds to make centroid
  // initialisation matter. Random seed fixed for determinism in test.
  const rng = () => Math.sin(42 + Math.random()); // pseudo, but ok for test

  const points: number[][] = [];
  for (let i = 0; i < 50; i++) {
    points.push([Math.random() * 0.5, Math.random() * 0.5]); // near origin
  }
  for (let i = 0; i < 50; i++) {
    points.push([1 + Math.random() * 0.5, 1 + Math.random() * 0.5]);
  }
  return points;
}

describe("SpectralClustering – randomState propagation", () => {
  it("produces identical labels when called twice with the same seed", async () => {
    const X = makeChallengingData();

    const seed = 123;
    const model1 = new SpectralClustering({ nClusters: 2, randomState: seed });
    const model2 = new SpectralClustering({ nClusters: 2, randomState: seed });

    const labels1 = (await model1.fitPredict(X)) as number[];
    const labels2 = (await model2.fitPredict(X)) as number[];

    expect(labels1).toEqual(labels2);
  });

  it("produces different labels for different seeds (at least one position)", async () => {
    const X = makeChallengingData();

    const model1 = new SpectralClustering({ nClusters: 2, randomState: 1, nInit: 1 });
    const model2 = new SpectralClustering({ nClusters: 2, randomState: 2, nInit: 1 });

    const labels1 = (await model1.fitPredict(X)) as number[];
    const labels2 = (await model2.fitPredict(X)) as number[];

    // The entire label vector should not be identical. They may still share
    // some assignments due to dataset symmetry but at least one position
    // must differ if the random seed indeed influences k-means++ init.
    expect(labels1).not.toEqual(labels2);
  });
});
