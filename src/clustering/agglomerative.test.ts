import fs from "fs";
import path from "path";

import { AgglomerativeClustering, AgglomerativeClusteringParams } from "..";

const COSINE_FIXTURE_DIR = path.join(process.cwd(), "__fixtures__", "agglomerative");

/** Bijective labelling equivalence (cluster ids may be permuted). */
function labelings_equivalent(a: number[], b: number[]): boolean {
  if (a.length !== b.length) return false;
  const fwd = new Map<number, number>();
  const rev = new Map<number, number>();
  for (let i = 0; i < a.length; i++) {
    if (fwd.has(a[i])) {
      if (fwd.get(a[i]) !== b[i]) return false;
    } else fwd.set(a[i], b[i]);
    if (rev.has(b[i])) {
      if (rev.get(b[i]) !== a[i]) return false;
    } else rev.set(b[i], a[i]);
  }
  return true;
}

describe("AgglomerativeClustering – cosine metric parity with scikit-learn", () => {
  const files = fs
    .readdirSync(COSINE_FIXTURE_DIR)
    .filter((f) => f.endsWith("_cosine.json") && !f.startsWith("medoids_"));

  it("has cosine reference fixtures", () => {
    expect(files.length).toBeGreaterThan(0);
  });

  for (const file of files) {
    const fixture = JSON.parse(
      fs.readFileSync(path.join(COSINE_FIXTURE_DIR, file), "utf-8"),
    ) as {
      X: number[][];
      params: { n_clusters: number; linkage: string; metric: string };
      labels: number[];
    };

    it(`matches sklearn labels for ${file} (linkage=${fixture.params.linkage})`, async () => {
      const model = new AgglomerativeClustering({
        n_clusters: fixture.params.n_clusters,
        linkage: fixture.params.linkage as AgglomerativeClusteringParams["linkage"],
        metric: "cosine",
      });
      const ours = await model.fit_predict(fixture.X);
      expect(labelings_equivalent(ours, fixture.labels)).toBe(true);
    });
  }
});

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
