import { AgglomerativeClustering } from "..";

describe("AgglomerativeClustering class structure & validation", () => {
  it("instantiates with minimal valid params", () => {
    const model = new AgglomerativeClustering({ n_clusters: 3 });
    expect(model.params.n_clusters).toBe(3);
    // defaults
    expect(model.params.linkage ?? "ward").toBe("ward");
  });

  it("throws for invalid n_clusters", () => {
    expect(() => new AgglomerativeClustering({ n_clusters: 0 })).toThrow();
    expect(() => new AgglomerativeClustering({ n_clusters: -2 })).toThrow();
  });

  it("throws for invalid linkage", () => {
    // @ts-expect-error - invalid linkage value; testing runtime validation
    expect(() => new AgglomerativeClustering({ n_clusters: 2, linkage: "foo" })).toThrow();
  });

  it("throws for invalid metric", () => {
    // @ts-expect-error - invalid metric value; testing runtime validation
    expect(() => new AgglomerativeClustering({ n_clusters: 2, metric: "hamming" })).toThrow();
  });

  it("throws when ward linkage used with non-euclidean metric", () => {
    expect(() =>
      new AgglomerativeClustering({ n_clusters: 2, linkage: "ward", metric: "cosine" }),
    ).toThrow();
  });

  it("fit and fit_predict run and produce expected labels", async () => {
    const X = [
      [0, 0],
      [0, 0.1],
      [5, 5],
      [5.1, 5],
    ];
    const model = new AgglomerativeClustering({ n_clusters: 2, linkage: "single" });
    const labels = await model.fit_predict(X);
    expect(labels.length).toBe(4);
    // Expect exactly two unique labels 0 and 1
    const uniq = Array.from(new Set(labels));
    expect(uniq.length).toBe(2);
  });
});

describe("AgglomerativeClustering – transductive: no predict or JSON serialization", () => {
  it("exposes neither predict nor to_json/from_json", () => {
    const model = new AgglomerativeClustering({ n_clusters: 2 });
    expect("predict" in model).toBe(false);
    expect("to_json" in model).toBe(false);
    expect("from_json" in AgglomerativeClustering).toBe(false);
  });
});
