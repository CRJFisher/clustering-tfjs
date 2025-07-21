#!/usr/bin/env python3
"""
Detailed step-by-step comparison of our SpectralClustering implementation with sklearn.
This script runs both implementations and compares intermediate results at each step.
"""

import numpy as np
import json
import subprocess
import sys
import os
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster._spectral import spectral_embedding
import warnings

# Silence warnings for cleaner output
warnings.filterwarnings('ignore')

def load_fixture(fixture_name):
    """Load a fixture file."""
    path = f"test/fixtures/spectral/{fixture_name}"
    with open(path, 'r') as f:
        return json.load(f)

def run_our_implementation(fixture_name):
    """Run our implementation using Node.js and return intermediate results."""
    # Create a test script
    test_script = f"""
const tf = require('@tensorflow/tfjs-node');
const {{ SpectralClustering }} = require('./dist/clustering/spectral');
const fs = require('fs');

// Adjusted Rand Index implementation
function adjustedRandIndex(labelsA, labelsB) {{
  if (labelsA.length !== labelsB.length) {{
    throw new Error("Label arrays must have same length");
  }}

  const n = labelsA.length;
  const labelToIndexA = new Map();
  const labelToIndexB = new Map();

  let nextA = 0;
  let nextB = 0;

  const clustersA = labelsA.map(label => {{
    if (!labelToIndexA.has(label)) {{
      labelToIndexA.set(label, nextA++);
    }}
    return labelToIndexA.get(label);
  }});

  const clustersB = labelsB.map(label => {{
    if (!labelToIndexB.has(label)) {{
      labelToIndexB.set(label, nextB++);
    }}
    return labelToIndexB.get(label);
  }});

  const nClassesA = nextA;
  const nClassesB = nextB;

  // Compute contingency table
  const contingency = Array(nClassesA).fill(null).map(() => Array(nClassesB).fill(0));
  
  for (let i = 0; i < n; i++) {{
    contingency[clustersA[i]][clustersB[i]]++;
  }}

  // Sum over rows and columns
  const sumA = new Array(nClassesA).fill(0);
  const sumB = new Array(nClassesB).fill(0);

  for (let i = 0; i < nClassesA; i++) {{
    for (let j = 0; j < nClassesB; j++) {{
      sumA[i] += contingency[i][j];
      sumB[j] += contingency[i][j];
    }}
  }}

  // Compute index
  let sumNij2 = 0;
  let sumNi2 = 0;
  let sumNj2 = 0;

  for (let i = 0; i < nClassesA; i++) {{
    for (let j = 0; j < nClassesB; j++) {{
      const nij = contingency[i][j];
      sumNij2 += nij * (nij - 1) / 2;
    }}
    sumNi2 += sumA[i] * (sumA[i] - 1) / 2;
  }}

  for (let j = 0; j < nClassesB; j++) {{
    sumNj2 += sumB[j] * (sumB[j] - 1) / 2;
  }}

  const expectedIndex = sumNi2 * sumNj2 / (n * (n - 1) / 2);
  const maxIndex = (sumNi2 + sumNj2) / 2;
  const index = sumNij2;

  if (maxIndex === expectedIndex) {{
    return 1.0;
  }}

  return (index - expectedIndex) / (maxIndex - expectedIndex);
}}

async function test() {{
    const fixture = JSON.parse(fs.readFileSync('test/fixtures/spectral/{fixture_name}', 'utf8'));
    
    const X = tf.tensor2d(fixture.X);
    const yTrue = fixture.labels;
    const params = fixture.params;
    
    const modelParams = {{
        nClusters: params.nClusters,
        affinity: params.affinity,
        randomState: params.randomState
    }};
    
    // Only add gamma for RBF
    if (params.affinity === 'rbf' && params.gamma !== null && params.gamma !== undefined) {{
        modelParams.gamma = params.gamma;
    }}
    
    // Only add nNeighbors for k-NN
    if (params.affinity === 'nearest_neighbors' && params.nNeighbors !== null && params.nNeighbors !== undefined) {{
        modelParams.nNeighbors = params.nNeighbors;
    }}
    
    const model = new SpectralClustering(modelParams);
    
    // Capture intermediate results
    const debugInfo = {{
        fixture: '{fixture_name}',
        params: params
    }};
    
    // Override internal methods to capture intermediate results
    const originalFit = model.fit.bind(model);
    model.fit = async function(X) {{
        // Run original fit
        await originalFit(X);
        
        // Capture affinity matrix
        if (model.affinity_ && model.affinity_.shape) {{
            debugInfo.affinityShape = model.affinity_.shape;
            debugInfo.affinityNonZeros = await tf.count(tf.notEqual(model.affinity_, 0)).data();
            
            // Sample values from affinity matrix
            const affinityData = await model.affinity_.data();
            debugInfo.affinitySample = Array.from(affinityData.slice(0, 5));
        }}
        
        // Capture Laplacian info if available
        if (model.laplacian_) {{
            debugInfo.laplacianShape = model.laplacian_.shape;
            const laplacianData = await model.laplacian_.data();
            debugInfo.laplacianDiag = Array.from(laplacianData.filter((_, i) => i % (model.laplacian_.shape[0] + 1) === 0).slice(0, 5));
        }}
        
        // Capture embedding info
        if (model.embedding_) {{
            debugInfo.embeddingShape = model.embedding_.shape;
            const embeddingData = await model.embedding_.data();
            debugInfo.embeddingSample = Array.from(embeddingData.slice(0, 10));
        }}
        
        return this;
    }};
    
    const predictions = await model.fitPredict(X);
    const ari = adjustedRandIndex(predictions, yTrue);
    
    debugInfo.predictions = Array.from(predictions);
    debugInfo.ari = ari;
    
    console.log(JSON.stringify(debugInfo, null, 2));
    
    X.dispose();
    model.dispose();
}}

test().catch(err => {{
    console.error('Error:', err);
    process.exit(1);
}});
"""
    
    # Write test script
    with open('temp_test_spectral.js', 'w') as f:
        f.write(test_script)
    
    try:
        # Run the test
        result = subprocess.run(['node', 'temp_test_spectral.js'], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error running our implementation: {result.stderr}")
            return None
            
        return json.loads(result.stdout)
    finally:
        # Clean up
        if os.path.exists('temp_test_spectral.js'):
            os.remove('temp_test_spectral.js')

def analyze_differences(fixture_name):
    """Analyze differences between sklearn and our implementation."""
    print(f"\n{'='*80}")
    print(f"Analyzing {fixture_name}")
    print('='*80)
    
    # Load fixture
    fixture = load_fixture(fixture_name)
    X = np.array(fixture['X'])
    y_true = np.array(fixture['labels'])
    params = fixture['params']
    
    # Extract parameters
    n_clusters = params['nClusters']
    affinity = params['affinity']
    gamma = params.get('gamma', None)
    n_neighbors = params.get('nNeighbors', 10)
    random_state = params.get('randomState', 42)
    
    print(f"\nDataset info:")
    print(f"  Shape: {X.shape}")
    print(f"  Parameters: n_clusters={n_clusters}, affinity={affinity}")
    
    # Run sklearn step by step
    print("\n--- SKLEARN ANALYSIS ---")
    
    # 1. Compute affinity matrix
    if affinity == 'rbf':
        if gamma is None:
            gamma_value = 1.0 / X.shape[1]
        else:
            gamma_value = gamma
        affinity_matrix = rbf_kernel(X, gamma=gamma_value)
        print(f"  RBF gamma: {gamma_value}")
    else:
        affinity_matrix = kneighbors_graph(X, n_neighbors=n_neighbors, 
                                         mode='connectivity', include_self=True)
        affinity_matrix = 0.5 * (affinity_matrix + affinity_matrix.T)
        print(f"  k-NN neighbors: {n_neighbors}")
    
    # 2. Get spectral embedding
    maps = spectral_embedding(affinity_matrix, n_components=n_clusters, 
                            drop_first=False, random_state=random_state)
    
    # Handle sparse matrices
    if hasattr(affinity_matrix, 'nnz'):
        nnz = affinity_matrix.nnz
    else:
        nnz = np.count_nonzero(affinity_matrix)
    
    print(f"  Affinity matrix: shape={affinity_matrix.shape}, nnz={nnz}")
    print(f"  Embedding shape: {maps.shape}")
    print(f"  Embedding first 5 values: {maps.flatten()[:5]}")
    
    # Check for constant columns
    for i in range(maps.shape[1]):
        unique_vals = len(np.unique(np.round(maps[:, i], 10)))
        print(f"  Embedding col {i}: {unique_vals} unique values")
    
    # 3. Run full sklearn
    kwargs = {
        'n_clusters': n_clusters,
        'affinity': affinity,
        'random_state': random_state,
        'n_init': 10,
    }
    
    if affinity == 'rbf' and gamma is not None:
        kwargs['gamma'] = gamma
    elif affinity == 'nearest_neighbors':
        kwargs['n_neighbors'] = n_neighbors
        
    model = SpectralClustering(**kwargs)
    y_sklearn = model.fit_predict(X)
    ari_sklearn = adjusted_rand_score(y_true, y_sklearn)
    
    print(f"  Final ARI: {ari_sklearn:.6f}")
    
    # Run our implementation
    print("\n--- OUR IMPLEMENTATION ---")
    our_result = run_our_implementation(fixture_name)
    
    if our_result:
        print(f"  ARI: {our_result['ari']:.6f}")
        if 'embeddingShape' in our_result:
            print(f"  Embedding shape: {our_result['embeddingShape']}")
            print(f"  Embedding first 10 values: {our_result['embeddingSample']}")
        if 'affinityNonZeros' in our_result:
            print(f"  Affinity non-zeros: {our_result['affinityNonZeros']}")
        
        # Compare
        print(f"\n--- COMPARISON ---")
        print(f"  ARI difference: {abs(ari_sklearn - our_result['ari']):.6f}")
        
        if ari_sklearn >= 0.95 and our_result['ari'] < 0.95:
            print(f"  ⚠️  FAILING: sklearn gets {ari_sklearn:.3f} but we get {our_result['ari']:.3f}")
            
            # Detailed debugging for failures
            if affinity == 'nearest_neighbors' and 'blobs' not in fixture_name:
                print("\n  SPECIAL ANALYSIS: k-NN case that sklearn solves perfectly")
                print("  This suggests sklearn might be doing something special with the embedding")
    
    return {
        'sklearn_ari': ari_sklearn,
        'our_ari': our_result['ari'] if our_result else None,
        'difference': abs(ari_sklearn - (our_result['ari'] if our_result else 0))
    }

def main():
    """Run detailed comparison for key fixtures."""
    # Focus on the most interesting cases
    fixtures = [
        'blobs_n2_knn.json',     # We pass this
        'blobs_n2_rbf.json',     # We fail this (!)
        'circles_n2_knn.json',   # We pass this
        'circles_n2_rbf.json',   # We fail this
        'moons_n2_knn.json',     # We fail this
        'moons_n2_rbf.json',     # We fail this
    ]
    
    results = {}
    for fixture in fixtures:
        result = analyze_differences(fixture)
        results[fixture] = result
    
    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print('='*80)
    
    for fixture, result in results.items():
        if result['our_ari'] is not None:
            status = "✓ PASS" if result['our_ari'] >= 0.95 else "✗ FAIL"
            print(f"{status} {fixture}: sklearn={result['sklearn_ari']:.3f}, " + 
                  f"ours={result['our_ari']:.3f}, diff={result['difference']:.3f}")

if __name__ == "__main__":
    main()