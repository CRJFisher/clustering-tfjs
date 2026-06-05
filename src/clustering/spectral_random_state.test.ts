import { SpectralClustering } from "../index";
import { make_random_stream } from "../random";

/**
 * Unit test verifying that the `random_state` provided to `SpectralClustering`
 * is forwarded to the internal `KMeans` initialisation which makes the final
 * cluster labels fully reproducible.
 */

function make_challenging_data(): number[][] {
  // Generate points in two overlapping Gaussian clouds to make centroid
  // initialisation matter. Random seed fixed for determinism in test.
  const rng = make_random_stream(42);

  const points: number[][] = [];
  for (let i = 0; i < 50; i++) {
    points.push([rng.rand() * 0.5, rng.rand() * 0.5]); // near origin
  }
  for (let i = 0; i < 50; i++) {
    points.push([1 + rng.rand() * 0.5, 1 + rng.rand() * 0.5]);
  }
  return points;
}

describe("SpectralClustering – randomState propagation", () => {
  it("produces identical labels when called twice with the same seed", async () => {
    const X = make_challenging_data();

    const seed = 123;
    const model1 = new SpectralClustering({ n_clusters: 2, random_state: seed });
    const model2 = new SpectralClustering({ n_clusters: 2, random_state: seed });

    const labels1 = await model1.fit_predict(X);
    const labels2 = await model2.fit_predict(X);

    expect(labels1).toEqual(labels2);
  });

  it("produces different labels for different seeds (at least one position)", async () => {
    const X = make_challenging_data();

    const model1 = new SpectralClustering({ n_clusters: 2, random_state: 1, n_init: 1 });
    const model2 = new SpectralClustering({ n_clusters: 2, random_state: 2, n_init: 1 });

    const labels1 = await model1.fit_predict(X);
    const labels2 = await model2.fit_predict(X);

    // The entire label vector should not be identical. They may still share
    // some assignments due to dataset symmetry but at least one position
    // must differ if the random seed indeed influences k-means++ init.
    expect(labels1).not.toEqual(labels2);
  });
});
