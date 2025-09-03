/**
 * Self-Organizing Maps (SOM) Example
 * 
 * Demonstrates various features of the SOM implementation including:
 * - Different topologies (rectangular vs hexagonal)
 * - Initialization methods (random, linear, PCA)
 * - Visualization with U-matrix
 * - Quality metrics (quantization and topographic error)
 */

const { Clustering } = require('../dist/index.js');

// Generate sample data
function generateSampleData() {
    const data = [];
    
    // Three distinct clusters
    // Cluster 1: centered at (2, 2)
    for (let i = 0; i < 20; i++) {
        data.push([
            2 + Math.random() * 2 - 1,
            2 + Math.random() * 2 - 1
        ]);
    }
    
    // Cluster 2: centered at (8, 2)
    for (let i = 0; i < 20; i++) {
        data.push([
            8 + Math.random() * 2 - 1,
            2 + Math.random() * 2 - 1
        ]);
    }
    
    // Cluster 3: centered at (5, 8)
    for (let i = 0; i < 20; i++) {
        data.push([
            5 + Math.random() * 2 - 1,
            8 + Math.random() * 2 - 1
        ]);
    }
    
    return data;
}

async function basicSOMExample() {
    console.log('=== Basic SOM Example ===\n');
    
    const data = generateSampleData();
    console.log(`Generated ${data.length} data points in 3 clusters\n`);
    
    // Create a 5x5 SOM
    const som = new Clustering.SOM({
        gridWidth: 5,
        gridHeight: 5,
        nClusters: 25,
        topology: 'rectangular',
        neighborhood: 'gaussian',
        initialization: 'pca',
        learningRate: 0.5,
        radius: 2.5,
        numEpochs: 100,
        randomState: 42
    });
    
    // Train the SOM
    console.log('Training SOM...');
    const labels = await som.fitPredict(data);
    
    // Display results
    console.log('Sample labels (first 10):', labels.slice(0, 10));
    
    // Get the weight vectors
    const weights = som.getWeights();
    console.log(`\nWeight vectors shape: [${weights.shape}]`);
    
    // Calculate quality metrics
    const quantError = som.quantizationError();
    const topoError = await som.topographicError(data);
    
    console.log(`\nQuality Metrics:`);
    console.log(`  Quantization Error: ${quantError.toFixed(4)}`);
    console.log(`  Topographic Error: ${topoError.toFixed(4)}`);
    
    // Clean up
    weights.dispose();
}

async function compareTopologies() {
    console.log('\n=== Comparing Topologies ===\n');
    
    const data = generateSampleData();
    
    // Test rectangular topology
    const rectSom = new Clustering.SOM({
        gridWidth: 4,
        gridHeight: 4,
        nClusters: 16,
        topology: 'rectangular',
        initialization: 'pca',
        numEpochs: 100,
        randomState: 42
    });
    
    await rectSom.fit(data);
    const rectQuantError = rectSom.quantizationError();
    const rectTopoError = await rectSom.topographicError(data);
    
    console.log('Rectangular Topology:');
    console.log(`  Quantization Error: ${rectQuantError.toFixed(4)}`);
    console.log(`  Topographic Error: ${rectTopoError.toFixed(4)}`);
    
    // Test hexagonal topology
    const hexSom = new Clustering.SOM({
        gridWidth: 4,
        gridHeight: 4,
        nClusters: 16,
        topology: 'hexagonal',
        initialization: 'pca',
        numEpochs: 100,
        randomState: 42
    });
    
    await hexSom.fit(data);
    const hexQuantError = hexSom.quantizationError();
    const hexTopoError = await hexSom.topographicError(data);
    
    console.log('\nHexagonal Topology:');
    console.log(`  Quantization Error: ${hexQuantError.toFixed(4)}`);
    console.log(`  Topographic Error: ${hexTopoError.toFixed(4)}`);
    
    // Compare results
    console.log('\nComparison:');
    if (hexTopoError < rectTopoError) {
        console.log('  ✓ Hexagonal topology preserves data structure better');
    } else {
        console.log('  ✓ Rectangular topology preserves data structure better');
    }
}

async function visualizationExample() {
    console.log('\n=== U-Matrix Visualization ===\n');
    
    const data = generateSampleData();
    
    // Create a smaller SOM for easier visualization
    const som = new Clustering.SOM({
        gridWidth: 3,
        gridHeight: 3,
        nClusters: 9,
        topology: 'hexagonal',
        neighborhood: 'gaussian',
        initialization: 'pca',
        numEpochs: 100
    });
    
    await som.fit(data);
    
    // Get U-matrix
    const uMatrix = som.getUMatrix();
    const uMatrixArray = await uMatrix.array();
    
    console.log('U-Matrix (distances between neighboring neurons):');
    console.log('Higher values indicate cluster boundaries\n');
    
    // Format U-matrix for display
    for (let i = 0; i < uMatrixArray.length; i++) {
        let row = '';
        for (let j = 0; j < uMatrixArray[i].length; j++) {
            row += uMatrixArray[i][j].toFixed(3) + '  ';
        }
        console.log(row);
    }
    
    // Clean up
    uMatrix.dispose();
}

async function compareInitializations() {
    console.log('\n=== Comparing Initialization Methods ===\n');
    
    const data = generateSampleData();
    const params = {
        gridWidth: 4,
        gridHeight: 4,
        nClusters: 16,
        topology: 'rectangular',
        numEpochs: 50,
        randomState: 42
    };
    
    // Test random initialization
    const randomSom = new Clustering.SOM({
        ...params,
        initialization: 'random'
    });
    await randomSom.fit(data);
    const randomError = randomSom.quantizationError();
    console.log(`Random initialization - Quantization Error: ${randomError.toFixed(4)}`);
    
    // Test linear initialization
    const linearSom = new Clustering.SOM({
        ...params,
        initialization: 'linear'
    });
    await linearSom.fit(data);
    const linearError = linearSom.quantizationError();
    console.log(`Linear initialization - Quantization Error: ${linearError.toFixed(4)}`);
    
    // Test PCA initialization
    const pcaSom = new Clustering.SOM({
        ...params,
        initialization: 'pca'
    });
    await pcaSom.fit(data);
    const pcaError = pcaSom.quantizationError();
    console.log(`PCA initialization - Quantization Error: ${pcaError.toFixed(4)}`);
    
    // Find best
    const errors = {
        'Random': randomError,
        'Linear': linearError,
        'PCA': pcaError
    };
    
    const best = Object.entries(errors).reduce((a, b) => 
        errors[a[0]] < errors[b[0]] ? a : b
    );
    
    console.log(`\n✓ Best initialization method: ${best[0]}`);
}

async function main() {
    try {
        console.log('Self-Organizing Maps (SOM) Examples\n');
        console.log('=====================================\n');
        
        // Initialize the library
        await Clustering.init();
        console.log(`Platform: ${Clustering.platform}`);
        console.log(`Backend: ${Clustering.features.backend}\n`);
        
        // Run examples
        await basicSOMExample();
        await compareTopologies();
        await visualizationExample();
        await compareInitializations();
        
        console.log('\n=====================================');
        console.log('✅ All SOM examples completed successfully!');
        
    } catch (error) {
        console.error('❌ Error:', error.message);
        console.error(error.stack);
        process.exit(1);
    }
}

// Run the examples
main();