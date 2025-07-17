import { SpectralClustering } from "../../src";

/**
 * Unit test verifying that the `randomState` provided to `SpectralClustering`
 * is forwarded to the internal `KMeans` initialisation which makes the final
 * cluster labels fully reproducible.
 */

function makeTwoBlobs(): number[][] {
  // Very obvious separation – four points near (0,0) and four near (5,5).
  return [
    [-0.2, 0.1],
    [0.1, -0.1],
    [0.2, 0.2],
    [-0.1, -0.2],
    [5.0, 5.1],
    [5.2, 4.9],
    [4.8, 5.0],
    [5.1, 5.2],
  ];
}

describe("SpectralClustering – randomState propagation", () => {
  it("produces identical labels when called twice with the same seed", async () => {
    const X = makeTwoBlobs();

    const seed = 123;
    const model1 = new SpectralClustering({ nClusters: 2, randomState: seed });
    const model2 = new SpectralClustering({ nClusters: 2, randomState: seed });

    const labels1 = (await model1.fitPredict(X)) as number[];
    const labels2 = (await model2.fitPredict(X)) as number[];

    expect(labels1).toEqual(labels2);
  });

  it("produces different labels for different seeds (at least one position)", async () => {
    const X = makeTwoBlobs();

    const model1 = new SpectralClustering({ nClusters: 2, randomState: 1 });
    const model2 = new SpectralClustering({ nClusters: 2, randomState: 2 });

    const labels1 = (await model1.fitPredict(X)) as number[];
    const labels2 = (await model2.fitPredict(X)) as number[];

    // The entire label vector should not be identical. They may still share
    // some assignments due to dataset symmetry but at least one position
    // must differ if the random seed indeed influences k-means++ init.
    expect(labels1).not.toEqual(labels2);
  });
});
