import fs from "fs";
import path from "path";

import { SpectralClustering } from "../../src";
import type { IntermediateSteps } from "../../src/clustering/spectral";

const FIXTURE_DIR = path.join(process.cwd(), "test", "fixtures", "spectral");

interface EmbeddingFixture {
  X: number[][];
  params: {
    nClusters: number;
    affinity: string;
    gamma: number;
    randomState: number;
  };
  labels: number[];
  embedding: number[][];
  eigenvalues: number[];
}

function disposeIntermediateSteps(steps: IntermediateSteps): void {
  steps.affinity.dispose();
  steps.laplacian.laplacian.dispose();
  steps.embedding.embedding.dispose();
  steps.embedding.eigenvalues.dispose();
  if (steps.embedding.rawEigenvectors) steps.embedding.rawEigenvectors.dispose();
  if (steps.laplacian.degrees) steps.laplacian.degrees.dispose();
  if (steps.laplacian.sqrtDegrees) steps.laplacian.sqrtDegrees.dispose();
}

function loadFixture(name: string): EmbeddingFixture | null {
  const p = path.join(FIXTURE_DIR, name);
  if (!fs.existsSync(p)) return null;
  return JSON.parse(fs.readFileSync(p, "utf-8"));
}

/**
 * Compute the Gram matrix G = U * U^T, which is invariant to rotations
 * within degenerate eigenspaces.
 */
function gramMatrix(emb: number[][]): number[][] {
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
  const blobsFixture = loadFixture("embedding_blobs_n3_rbf.json");

  if (!blobsFixture) {
    it("skipped – blobs embedding fixture not found", () => {
      expect(true).toBe(true);
    });
  } else {
    it("embedding subspace matches sklearn (Gram matrix comparison, blobs)", async () => {
      const model = new SpectralClustering({
        nClusters: blobsFixture.params.nClusters,
        affinity: blobsFixture.params.affinity as "rbf",
        gamma: blobsFixture.params.gamma,
        randomState: blobsFixture.params.randomState,
      });

      const steps = await model.fitWithIntermediateSteps(blobsFixture.X);
      const ourEmb = (await steps.embedding.embedding.array()) as number[][];
      const skEmb = blobsFixture.embedding;
      const n = blobsFixture.X.length;

      expect(ourEmb.length).toBe(n);
      expect(ourEmb[0].length).toBe(blobsFixture.params.nClusters);

      // Compare Gram matrices: G_ours[i][j] ≈ G_sklearn[i][j]
      // This is rotation-invariant within degenerate eigenspaces
      const G_ours = gramMatrix(ourEmb);
      const G_sklearn = gramMatrix(skEmb);

      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          expect(G_ours[i][j]).toBeCloseTo(G_sklearn[i][j], 2);
        }
      }

      disposeIntermediateSteps(steps);
      model.dispose();
    }, 20000);
  }

  // --- Test 2: Column-wise match (moons, distinct eigenvalues) ---
  const moonsFixture = loadFixture("embedding_moons_n2_rbf.json");

  if (!moonsFixture) {
    it("skipped – moons embedding fixture not found", () => {
      expect(true).toBe(true);
    });
  } else {
    it("embedding columns match sklearn per-column with sign alignment (moons)", async () => {
      const model = new SpectralClustering({
        nClusters: moonsFixture.params.nClusters,
        affinity: moonsFixture.params.affinity as "rbf",
        gamma: moonsFixture.params.gamma,
        randomState: moonsFixture.params.randomState,
      });

      const steps = await model.fitWithIntermediateSteps(moonsFixture.X);
      const ourEmb = (await steps.embedding.embedding.array()) as number[][];
      const skEmb = moonsFixture.embedding;
      const nClusters = moonsFixture.params.nClusters;
      const n = moonsFixture.X.length;

      expect(ourEmb.length).toBe(n);
      expect(ourEmb[0].length).toBe(nClusters);

      for (let col = 0; col < nClusters; col++) {
        const ourCol = ourEmb.map((row) => row[col]);
        const refCol = skEmb.map((row) => row[col]);

        // Cosine similarity (handles sign ambiguity)
        let dot = 0,
          normOur = 0,
          normRef = 0;
        for (let i = 0; i < n; i++) {
          dot += ourCol[i] * refCol[i];
          normOur += ourCol[i] * ourCol[i];
          normRef += refCol[i] * refCol[i];
        }
        const cosine = dot / (Math.sqrt(normOur) * Math.sqrt(normRef));

        // Should be +1 or -1
        expect(Math.abs(cosine)).toBeGreaterThan(0.99);

        // Element-wise after sign alignment
        const sign = cosine > 0 ? 1 : -1;
        for (let i = 0; i < n; i++) {
          expect(sign * ourCol[i]).toBeCloseTo(refCol[i], 2);
        }
      }

      disposeIntermediateSteps(steps);
      model.dispose();
    }, 20000);
  }

  // --- Test 3: No NaN/Inf in embeddings ---
  if (blobsFixture) {
    it("embedding has no NaN or Inf values", async () => {
      const model = new SpectralClustering({
        nClusters: blobsFixture.params.nClusters,
        affinity: blobsFixture.params.affinity as "rbf",
        gamma: blobsFixture.params.gamma,
        randomState: blobsFixture.params.randomState,
      });

      const steps = await model.fitWithIntermediateSteps(blobsFixture.X);
      const data = await steps.embedding.embedding.data();

      for (const v of data) {
        expect(isFinite(v)).toBe(true);
        expect(isNaN(v)).toBe(false);
      }

      disposeIntermediateSteps(steps);
      model.dispose();
    }, 20000);
  }
});
