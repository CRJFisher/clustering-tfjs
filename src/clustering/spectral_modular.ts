import type {
  DataMatrix,
  LabelVector,
  SpectralClusteringParams,
  BaseClustering,
} from "./types";
import * as tf from "../utils/tensorflow";

export interface LaplacianResult {
  laplacian: tf.Tensor2D;
  degrees?: tf.Tensor1D;
  sqrtDegrees?: tf.Tensor1D;
}

export interface EmbeddingResult {
  embedding: tf.Tensor2D;
  eigenvalues: tf.Tensor1D;
  rawEigenvectors?: tf.Tensor2D;
  scalingFactors?: tf.Tensor1D;
}

export interface DebugInfo {
  affinityStats?: {
    shape: number[];
    nnz: number;
    min: number;
    max: number;
    mean: number;
  };
  laplacianSpectrum?: number[];
  embeddingStats?: {
    shape: number[];
    uniqueValuesPerDim: number[];
    scalingFactors?: number[];
  };
  clusteringMetrics?: {
    inertia: number;
    iterations: number;
  };
}

/**
 * Modular SpectralClustering implementation that exposes intermediate steps.
 * This allows for better testing, debugging, and comparison with reference implementations.
 */
export class SpectralClusteringModular
  implements BaseClustering<SpectralClusteringParams>
{
  public readonly params: SpectralClusteringParams;
  public labels_: number[] | null = null;
  public affinityMatrix_: tf.Tensor2D | null = null;
  
  private debugInfo_: DebugInfo | null = null;
  private captureDebugInfo: boolean;

  constructor(params: SpectralClusteringParams & { captureDebugInfo?: boolean }) {
    const { captureDebugInfo = false, ...clusteringParams } = params;
    this.params = { ...clusteringParams };
    this.captureDebugInfo = captureDebugInfo;
    this.validateParams();
  }

  /**
   * Get debug information if captureDebugInfo was enabled.
   */
  public getDebugInfo(): DebugInfo | null {
    return this.debugInfo_;
  }

  /**
   * Step 1: Compute affinity matrix from input data.
   * This is exposed as a public method for testing and debugging.
   */
  public async computeAffinityMatrix(X: DataMatrix): Promise<tf.Tensor2D> {
    const Xtensor = X instanceof tf.Tensor
      ? (tf.cast(X as tf.Tensor2D, "float32") as tf.Tensor2D)
      : tf.tensor2d(X as number[][], undefined, "float32");

    const affinity = SpectralClusteringModular.computeAffinityMatrix(Xtensor, this.params);
    
    if (this.captureDebugInfo) {
      const data = await affinity.data();
      const dataArray = Array.from(data);
      const nnz = dataArray.filter((v: number) => v !== 0).length;
      this.debugInfo_ = {
        ...this.debugInfo_,
        affinityStats: {
          shape: affinity.shape,
          nnz,
          min: Math.min(...dataArray),
          max: Math.max(...dataArray),
          mean: dataArray.reduce((a: number, b: number) => a + b, 0) / dataArray.length,
        },
      };
    }

    return affinity;
  }

  /**
   * Step 2: Compute normalized Laplacian from affinity matrix.
   * Returns both the Laplacian and auxiliary information.
   */
  public async computeLaplacian(affinity: tf.Tensor2D): Promise<LaplacianResult> {
    const { normalised_laplacian } = await import("../utils/laplacian");
    
    // For now, just compute the basic Laplacian
    const laplacian = tf.tidy(() => normalised_laplacian(affinity));
    
    if (this.captureDebugInfo) {
      // Compute eigenvalues for spectrum analysis
      const { jacobi_eigen_decomposition } = await import("../utils/laplacian");
      const { eigenvalues } = await jacobi_eigen_decomposition(laplacian);
      // eigenvalues is an array, take first 10
      const spectrum = eigenvalues.slice(0, Math.min(10, eigenvalues.length));
      
      this.debugInfo_ = {
        ...this.debugInfo_,
        laplacianSpectrum: spectrum,
      };
    }

    return { laplacian };
  }

  /**
   * Step 3: Compute spectral embedding from Laplacian.
   * Returns the embedding and associated eigenvalues.
   */
  public async computeSpectralEmbedding(
    laplacian: tf.Tensor2D
  ): Promise<EmbeddingResult> {
    const { smallest_eigenvectors_with_values } = await import(
      "../utils/smallest_eigenvectors_with_values"
    );

    const { eigenvectors: U_full, eigenvalues } = smallest_eigenvectors_with_values(
      laplacian,
      this.params.nClusters
    );

    // Apply diffusion map scaling: scale by sqrt(1 - eigenvalue)
    const embedding = tf.tidy(() => {
      const eigenvals = tf.slice(eigenvalues, [0], [this.params.nClusters]) as tf.Tensor1D;
      const scalingFactors = tf.sqrt(
        tf.maximum(tf.scalar(0), tf.sub(tf.scalar(1), eigenvals))
      ) as tf.Tensor1D;
      
      const scalingFactors2D = scalingFactors.reshape([1, -1]) as tf.Tensor2D;
      const U_selected = tf.slice(U_full, [0, 0], [-1, this.params.nClusters]) as tf.Tensor2D;
      
      return U_selected.mul(scalingFactors2D) as tf.Tensor2D;
    });

    if (this.captureDebugInfo) {
      const embData = await embedding.data();
      const [n, k] = embedding.shape;
      const uniqueValuesPerDim: number[] = [];
      
      for (let i = 0; i < k; i++) {
        const col = embData.slice(i * n, (i + 1) * n);
        const unique = new Set(col.map(v => Math.round(v * 1e10) / 1e10));
        uniqueValuesPerDim.push(unique.size);
      }

      const scalingFactors = tf.slice(eigenvalues, [0], [this.params.nClusters]) as tf.Tensor1D;
      const scalingData = await scalingFactors.data();
      
      this.debugInfo_ = {
        ...this.debugInfo_,
        embeddingStats: {
          shape: embedding.shape,
          uniqueValuesPerDim,
          scalingFactors: Array.from(scalingData),
        },
      };
      
      scalingFactors.dispose();
    }

    const result: EmbeddingResult = {
      embedding,
      eigenvalues: tf.slice(eigenvalues, [0], [this.params.nClusters]) as tf.Tensor1D,
    };

    // Clean up
    U_full.dispose();
    
    return result;
  }

  /**
   * Step 4: Perform clustering on the embedding.
   * Returns cluster labels.
   */
  public async performClustering(embedding: tf.Tensor2D): Promise<number[]> {
    const { KMeans } = await import("./kmeans");
    
    const kmParams = {
      nClusters: this.params.nClusters,
      randomState: this.params.randomState,
      nInit: this.params.nInit ?? 10,
    } as const;

    const km = new KMeans(kmParams);
    await km.fit(embedding);
    
    if (this.captureDebugInfo && km.inertia_ !== null) {
      this.debugInfo_ = {
        ...this.debugInfo_,
        clusteringMetrics: {
          inertia: km.inertia_,
          iterations: 0, // KMeans doesn't expose iteration count currently
        },
      };
    }

    return km.labels_ as number[];
  }

  /**
   * Main pipeline that orchestrates all steps.
   */
  public async fit(X: DataMatrix): Promise<void> {
    // Reset state
    this.dispose();
    this.debugInfo_ = this.captureDebugInfo ? {} : null;

    // Step 1: Affinity matrix
    this.affinityMatrix_ = await this.computeAffinityMatrix(X);
    
    const affinitySum = (await this.affinityMatrix_.sum().data())[0];
    if (affinitySum === 0) {
      throw new Error(
        "Affinity matrix contains only zeros – cannot perform spectral clustering."
      );
    }

    // Step 2: Laplacian
    const { laplacian } = await this.computeLaplacian(this.affinityMatrix_);

    // Step 3: Spectral embedding
    const { embedding, eigenvalues } = await this.computeSpectralEmbedding(laplacian);

    // Step 4: Clustering
    this.labels_ = await this.performClustering(embedding);

    // Clean up
    laplacian.dispose();
    embedding.dispose();
    eigenvalues.dispose();
  }

  public async fitPredict(X: DataMatrix): Promise<LabelVector> {
    await this.fit(X);
    return this.labels_ as number[];
  }

  public dispose(): void {
    if (this.affinityMatrix_) {
      this.affinityMatrix_.dispose();
      this.affinityMatrix_ = null;
    }
    this.labels_ = null;
    this.debugInfo_ = null;
  }

  private validateParams(): void {
    if (this.params.nClusters < 2) {
      throw new Error("nClusters must be at least 2.");
    }

    if (this.params.affinity === "rbf") {
      if (this.params.gamma !== undefined && this.params.gamma <= 0) {
        throw new Error("gamma must be positive for RBF affinity.");
      }
    } else if (this.params.affinity === "nearest_neighbors") {
      if (this.params.nNeighbors !== undefined && this.params.nNeighbors < 1) {
        throw new Error("nNeighbors must be at least 1.");
      }
    }
  }

  /**
   * Static method for affinity computation.
   * Delegates to the original implementation for consistency.
   */
  private static computeAffinityMatrix(
    X: tf.Tensor2D,
    params: SpectralClusteringParams
  ): tf.Tensor2D {
    // Import the original implementation to avoid code duplication
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { SpectralClustering } = require("./spectral") as typeof import("./spectral");
    return (SpectralClustering as unknown as { computeAffinityMatrix: (X: tf.Tensor2D, params: SpectralClusteringParams) => tf.Tensor2D }).computeAffinityMatrix(X, params);
  }
}

// Re-export the original SpectralClustering for backward compatibility
export { SpectralClustering } from "./spectral";