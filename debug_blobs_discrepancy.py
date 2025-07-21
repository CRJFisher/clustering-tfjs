#!/usr/bin/env python3
"""
Debug the discrepancy between test results and Python comparison for blobs datasets.
"""

import numpy as np
import json
import subprocess
import os

def load_fixture(fixture_name):
    """Load a fixture file."""
    path = f"test/fixtures/spectral/{fixture_name}"
    with open(path, 'r') as f:
        return json.load(f)

def run_nodejs_test(fixture_name):
    """Run a single test using Node.js test runner."""
    test_script = f"""
const tf = require('@tensorflow/tfjs-node');
const {{ SpectralClustering }} = require('./dist/clustering/spectral');
const fs = require('fs');

// Adjusted Rand Index
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
    
    // Test with src version
    const modelSrc = new SpectralClustering({{
        nClusters: fixture.params.nClusters,
        affinity: fixture.params.affinity,
        gamma: fixture.params.affinity === 'rbf' ? fixture.params.gamma : undefined,
        nNeighbors: fixture.params.affinity === 'nearest_neighbors' ? fixture.params.nNeighbors : undefined,
        randomState: fixture.params.randomState
    }});
    
    const X = tf.tensor2d(fixture.X);
    const predSrc = await modelSrc.fitPredict(X);
    const ariSrc = adjustedRandIndex(predSrc, fixture.labels);
    
    console.log(JSON.stringify({{
        fixture: '{fixture_name}',
        ariSrc: ariSrc,
        predSrc: Array.from(predSrc).slice(0, 10),
        labels: fixture.labels.slice(0, 10)
    }}, null, 2));
    
    X.dispose();
    modelSrc.dispose();
}}

test().catch(err => {{
    console.error('Error:', err);
    process.exit(1);
}});
"""
    
    # Write test script
    with open('temp_debug_test.js', 'w') as f:
        f.write(test_script)
    
    try:
        # Run the test
        result = subprocess.run(['node', 'temp_debug_test.js'], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return None
            
        return json.loads(result.stdout)
    finally:
        if os.path.exists('temp_debug_test.js'):
            os.remove('temp_debug_test.js')

def main():
    """Debug blobs datasets."""
    fixtures = ['blobs_n2_knn.json', 'blobs_n2_rbf.json']
    
    for fixture_name in fixtures:
        print(f"\n{'='*60}")
        print(f"Testing {fixture_name}")
        print('='*60)
        
        # Load fixture
        fixture = load_fixture(fixture_name)
        print(f"Fixture labels (first 10): {fixture['labels'][:10]}")
        
        # Run src test
        result = run_nodejs_test(fixture_name)
        if result:
            print(f"Source ARI: {result['ariSrc']:.6f}")
            print(f"Source predictions (first 10): {result['predSrc']}")
            print(f"Expected labels (first 10): {result['labels']}")
            
            # Compare with expected sklearn labels
            if 'sklearnLabels' in fixture:
                print(f"Sklearn labels (first 10): {fixture['sklearnLabels'][:10]}")

if __name__ == "__main__":
    main()