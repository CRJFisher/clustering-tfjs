import fs from "fs";
import path from "path";

import { SpectralClustering } from "../../src";
import type { IntermediateSteps } from "../../src/clustering/spectral";

const FIXTURE_DIR = path.join(process.cwd(), "test", "fixtures", "spectral");

interface EmbeddingFixture {
  X: number[][];
  params: {
    n_clusters: number;
    affinity: string;
    gamma: number;
    random_state: number;
  };
  labels: number[];
  embedding: number[][];
  eigenvalues: number[];
}

function dispose_intermediate_steps(steps: IntermediateSteps): void {
  steps.affinity.dispose();
  steps.laplacian.laplacian.dispose();
  steps.embedding.embedding.dispose();
  steps.embedding.eigenvalues.dispose();
  if (steps.embedding.raw_eigenvectors) steps.embedding.raw_eigenvectors.dispose();
  if (steps.laplacian.degrees) steps.laplacian.degrees.dispose();
  if (steps.laplacian.sqrt_degrees) steps.laplacian.sqrt_degrees.dispose();
}

function load_fixture(name: string): EmbeddingFixture | null {
  const p = path.join(FIXTURE_DIR, name);
  if (!fs.existsSync(p)) return null;
  return JSON.parse(fs.readFileSync(p, "utf-8"));
}

/**
 * Compute the Gram matrix G = U * U^T, which is invariant to rotations
 * within degenerate eigenspaces.
 */
function gram_matrix(emb: number[][]): number[][] {
  const n = emb.length;
  const G: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      let dot = 0;
      for (let k = 0; k < emb[0].length; k++) {
        dot += emb[i][k] * emb[j][k];
      }
      G[i][j] = dot;
      G[j][i] = dot;
    }
  }
  return G;
}

describe("SpectralClustering – numerical embedding match with sklearn", () => {
  // --- Test 1: Subspace match via Gram matrix (blobs, degenerate eigenvalues) ---
  const blobs_fixture = load_fixture("embedding_blobs_n3_rbf.json");

  if (!blobs_fixture) {
    it("skipped – blobs embedding fixture not found", () => {
      expect(true).toBe(true);
    });
  } else {
    it("embedding subspace matches sklearn (Gram matrix comparison, blobs)", async () => {
      const model = new SpectralClustering({
        n_clusters: blobs_fixture.params.n_clusters,
        affinity: blobs_fixture.params.affinity as "rbf",
        gamma: blobs_fixture.params.gamma,
        random_state: blobs_fixture.params.random_state,
      });

      const steps = await model.fit_with_intermediate_steps(blobs_fixture.X);
      const our_emb = (await steps.embedding.embedding.array()) as number[][];
      const sk_emb = blobs_fixture.embedding;
      const n = blobs_fixture.X.length;

      expect(our_emb.length).toBe(n);
      expect(our_emb[0].length).toBe(blobs_fixture.params.n_clusters);

      // Compare Gram matrices: G_ours[i][j] ≈ G_sklearn[i][j]
      // This is rotation-invariant within degenerate eigenspaces
      const G_ours = gram_matrix(our_emb);
      const G_sklearn = gram_matrix(sk_emb);

      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          expect(G_ours[i][j]).toBeCloseTo(G_sklearn[i][j], 2);
        }
      }

      dispose_intermediate_steps(steps);
      model.dispose();
    }, 20000);
  }

  // --- Test 2: Column-wise match (moons, distinct eigenvalues) ---
  const moons_fixture = load_fixture("embedding_moons_n2_rbf.json");

  if (!moons_fixture) {
    it("skipped – moons embedding fixture not found", () => {
      expect(true).toBe(true);
    });
  } else {
    it("embedding columns match sklearn per-column with sign alignment (moons)", async () => {
      const model = new SpectralClustering({
        n_clusters: moons_fixture.params.n_clusters,
        affinity: moons_fixture.params.affinity as "rbf",
        gamma: moons_fixture.params.gamma,
        random_state: moons_fixture.params.random_state,
      });

      const steps = await model.fit_with_intermediate_steps(moons_fixture.X);
      const our_emb = (await steps.embedding.embedding.array()) as number[][];
      const sk_emb = moons_fixture.embedding;
      const n_clusters = moons_fixture.params.n_clusters;
      const n = moons_fixture.X.length;

      expect(our_emb.length).toBe(n);
      expect(our_emb[0].length).toBe(n_clusters);

      for (let col = 0; col < n_clusters; col++) {
        const our_col = our_emb.map((row) => row[col]);
        const ref_col = sk_emb.map((row) => row[col]);

        // Cosine similarity (handles sign ambiguity)
        let dot = 0,
          norm_our = 0,
          norm_ref = 0;
        for (let i = 0; i < n; i++) {
          dot += our_col[i] * ref_col[i];
          norm_our += our_col[i] * our_col[i];
          norm_ref += ref_col[i] * ref_col[i];
        }
        const cosine = dot / (Math.sqrt(norm_our) * Math.sqrt(norm_ref));

        // Should be +1 or -1
        expect(Math.abs(cosine)).toBeGreaterThan(0.99);

        // Element-wise after sign alignment
        const sign = cosine > 0 ? 1 : -1;
        for (let i = 0; i < n; i++) {
          expect(sign * our_col[i]).toBeCloseTo(ref_col[i], 2);
        }
      }

      dispose_intermediate_steps(steps);
      model.dispose();
    }, 20000);
  }

  // --- Test 3: No NaN/Inf in embeddings ---
  if (blobs_fixture) {
    it("embedding has no NaN or Inf values", async () => {
      const model = new SpectralClustering({
        n_clusters: blobs_fixture.params.n_clusters,
        affinity: blobs_fixture.params.affinity as "rbf",
        gamma: blobs_fixture.params.gamma,
        random_state: blobs_fixture.params.random_state,
      });

      const steps = await model.fit_with_intermediate_steps(blobs_fixture.X);
      const data = await steps.embedding.embedding.data();

      for (const v of data) {
        expect(isFinite(v)).toBe(true);
        expect(isNaN(v)).toBe(false);
      }

      dispose_intermediate_steps(steps);
      model.dispose();
    }, 20000);
  }
});
