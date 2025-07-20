import * as tf from '@tensorflow/tfjs-node';
import { SpectralClustering } from './src';
import fs from 'fs';

// Extend SpectralClustering to expose internal embedding
class SpectralClusteringDebug extends SpectralClustering {
  public lastEmbedding: number[][] | null = null;
  
  async fit(X: any): Promise<void> {
    // Call parent fit
    await super.fit(X);
    
    // Extract embedding by running through the pipeline again
    const Xtensor: tf.Tensor2D = X instanceof tf.Tensor
      ? (tf.cast(X as tf.Tensor2D, "float32") as tf.Tensor2D)
      : tf.tensor2d(X as number[][], undefined, "float32");
    
    console.log('DEBUG: Re-running pipeline to extract embedding...');
    
    // Get affinity matrix (already computed in parent)
    const affinity = this.affinityMatrix_!;
    
    // Get Laplacian
    const { normalised_laplacian, degree_vector } = await import("./src/utils/laplacian");
    const laplacian = normalised_laplacian(affinity);
    
    // Get eigenvectors with eigenvalues for diffusion scaling
    const { smallest_eigenvectors_with_values } = await import("./src/utils/smallest_eigenvectors_with_values");
    const { eigenvectors: U_full_raw, eigenvalues } = smallest_eigenvectors_with_values(
      laplacian, 
      this.params.nClusters
    );
    
    // Apply diffusion map scaling like sklearn does
    const U_full = tf.tidy(() => {
      // Compute scaling factors: sqrt(1 - eigenvalue)
      const scalingFactors = eigenvalues.sub(1).mul(-1).sqrt() as tf.Tensor1D;
      
      // Scale each column of eigenvectors
      const scaled = U_full_raw.mul(scalingFactors.expandDims(0)) as tf.Tensor2D;
      
      // Also apply the D^(-1/2) normalization
      const deg = degree_vector(this.affinityMatrix_ as tf.Tensor2D);
      const invSqrtDeg = tf.where(
        deg.equal(0),
        tf.zerosLike(deg),
        deg.pow(-0.5),
      ) as tf.Tensor1D;
      
      const result = scaled.div(invSqrtDeg.expandDims(1)) as tf.Tensor2D;
      deg.dispose();
      return result;
    });
    
    // Use the first nClusters columns
    const U = tf.slice(U_full, [0, 0], [-1, this.params.nClusters]) as tf.Tensor2D;
    
    // Extract embedding
    const Uarr = await U.array() as number[][];
    const nRows = Uarr.length;
    const nCols = Uarr[0].length;
    
    this.lastEmbedding = [];
    for (let i = 0; i < nRows; i++) {
      this.lastEmbedding.push(Uarr[i].slice(0, nCols));
    }
    
    // Check eigenvalues for debugging
    const eigVals = await eigenvalues.array() as number[];
    console.log('DEBUG: First few eigenvalues:', eigVals.slice(0, 5));
    console.log('DEBUG: Scaling factors:', eigVals.slice(0, 5).map(v => Math.sqrt(1 - v)));
    
    // Cleanup
    laplacian.dispose();
    U_full_raw.dispose();
    U_full.dispose();
    U.dispose();
    eigenvalues.dispose();
    if (!(X instanceof tf.Tensor)) {
      Xtensor.dispose();
    }
  }
}

async function extractEmbedding() {
  // Use circles_n2_knn as test case (ARI = 0.747)
  const fixture = require('./test/fixtures/spectral/circles_n2_knn.json');
  
  console.log('Extracting spectral embedding for circles_n2_knn');
  console.log('Expected ARI: 0.95+, Current ARI: 0.747\n');
  
  const model = new SpectralClusteringDebug({
    nClusters: fixture.params.nClusters,
    affinity: fixture.params.affinity,
    nNeighbors: fixture.params.nNeighbors,
    randomState: fixture.params.randomState
  });
  
  await model.fit(fixture.X);
  
  if (!model.lastEmbedding) {
    throw new Error('Failed to extract embedding');
  }
  
  console.log('Embedding shape:', model.lastEmbedding.length, 'x', model.lastEmbedding[0].length);
  
  // Save embedding for comparison
  const output = {
    dataset: 'circles_n2_knn',
    shape: [model.lastEmbedding.length, model.lastEmbedding[0].length],
    embedding: model.lastEmbedding,
    labels: model.labels_
  };
  
  fs.writeFileSync('our_embedding.json', JSON.stringify(output, null, 2));
  console.log('Saved embedding to our_embedding.json');
  
  // Show first few rows
  console.log('\nFirst 5 rows of embedding:');
  for (let i = 0; i < 5; i++) {
    console.log(`Row ${i}:`, model.lastEmbedding[i].map(v => v.toFixed(6)).join(', '));
  }
  
  // Check embedding properties
  console.log('\nEmbedding statistics:');
  const flat = model.lastEmbedding.flat();
  console.log('Min value:', Math.min(...flat).toFixed(6));
  console.log('Max value:', Math.max(...flat).toFixed(6));
  console.log('Mean value:', (flat.reduce((a, b) => a + b) / flat.length).toFixed(6));
  
  // Check row norms
  console.log('\nRow norms (first 5):');
  for (let i = 0; i < 5; i++) {
    const row = model.lastEmbedding[i];
    const norm = Math.sqrt(row.reduce((sum, val) => sum + val * val, 0));
    console.log(`Row ${i} norm:`, norm.toFixed(6));
  }
  
  model.dispose();
}

extractEmbedding().catch(console.error);