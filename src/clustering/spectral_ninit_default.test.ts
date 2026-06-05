import { SpectralClustering } from "../index";

describe("SpectralClustering – implicit n_init default", () => {
  it("uses n_init = 10 when the user does not provide one", async () => {
    const X = [
      [0, 0],
      [1, 1],
    ];

    const model = new SpectralClustering({ n_clusters: 2 });
    await model.fit(X);

    // The secret debug property is added by the implementation solely for
    // unit-testing / introspection purposes.
    const params = (model as SpectralClustering & {
      _debug_last_kmeans_params_: { n_init?: number };
    })._debug_last_kmeans_params_;
    expect(params).toBeDefined();
    expect(params.n_init).toBe(10);
  });
});

