import { SpectralClustering } from "..";

describe("SpectralClustering class structure & validation", () => {
  it("instantiates with minimal valid params (defaults)", () => {
    const model = new SpectralClustering({ n_clusters: 3 });
    expect(model.params.n_clusters).toBe(3);
    expect(model.params.affinity ?? "rbf").toBe("rbf");
  });

  it("accepts nearest_neighbors with n_neighbors", () => {
    const model = new SpectralClustering({
      n_clusters: 2,
      affinity: "nearest_neighbors",
      n_neighbors: 15,
    });
    expect(model.params.affinity).toBe("nearest_neighbors");
    expect(model.params.n_neighbors).toBe(15);
  });

  it("throws for invalid n_clusters", () => {
    expect(() => new SpectralClustering({ n_clusters: 0 })).toThrow();
  });

  it("throws for invalid affinity string", () => {
    expect(
      // @ts-expect-error - invalid affinity value; testing runtime validation
      () => new SpectralClustering({ n_clusters: 2, affinity: "foo" }),
    ).toThrow();
  });

  it("throws when gamma provided with nearest_neighbors", () => {
    expect(() =>
      new SpectralClustering({
        n_clusters: 2,
        affinity: "nearest_neighbors",
        gamma: 0.5,
      }),
    ).toThrow();
  });

  it("throws when n_neighbors provided with rbf affinity", () => {
    expect(() =>
      new SpectralClustering({ n_clusters: 2, n_neighbors: 5 }),
    ).toThrow();
  });
});


describe("SpectralClustering – transductive: no predict or JSON serialization", () => {
  it("exposes neither predict nor to_json/from_json", () => {
    const model = new SpectralClustering({ n_clusters: 2 });
    expect("predict" in model).toBe(false);
    expect("to_json" in model).toBe(false);
    expect("from_json" in SpectralClustering).toBe(false);
  });
});

describe("SpectralClustering – fit_predict output contract", () => {
  it("returns labels array of length n_samples", async () => {
    const X = [
      [0, 0], [0.1, 0.1], [0.2, 0],
      [5, 5], [5.1, 5.1], [5.2, 5],
      [10, 0], [10.1, 0.1], [10.2, 0],
    ];
    const sc = new SpectralClustering({ n_clusters: 3, random_state: 42 });
    const labels = await sc.fit_predict(X);
    expect(labels.length).toBe(X.length);
  });

  it("returns labels array of length n_samples via intensive_parameter_sweep", async () => {
    const X = [
      [0, 0], [0.1, 0.1], [0.2, 0],
      [5, 5], [5.1, 5.1], [5.2, 5],
      [10, 0], [10.1, 0.1], [10.2, 0],
    ];
    const sc = new SpectralClustering({
      n_clusters: 3,
      random_state: 42,
      intensive_parameter_sweep: true,
      affinity: 'rbf',
      gamma_range: [1.0],
    });
    const labels = await sc.fit_predict(X);
    expect(labels.length).toBe(X.length);
  });
});
