/**
 * Node.js GPU example for clustering-tfjs
 * 
 * Demonstrates how to use GPU acceleration with @tensorflow/tfjs-node-gpu.
 * Note: Requires @tensorflow/tfjs-node-gpu to be installed.
 */

const { Clustering } = require('../dist/index.js');

async function benchmark(name, fn) {
    const start = process.hrtime.bigint();
    const result = await fn();
    const end = process.hrtime.bigint();
    const ms = Number(end - start) / 1_000_000;
    console.log(`${name}: ${ms.toFixed(2)}ms`);
    return result;
}

async function main() {
    try {
        console.log('Clustering.js - Node.js GPU Example\n');
        
        // Check if GPU backend is available
        try {
            require.resolve('@tensorflow/tfjs-node-gpu');
            console.log('✅ GPU backend (@tensorflow/tfjs-node-gpu) is available');
        } catch (e) {
            console.log('❌ GPU backend not available. Install with:');
            console.log('   npm install @tensorflow/tfjs-node-gpu');
            console.log('\nFalling back to CPU backend...\n');
        }
        
        // Initialize
        await Clustering.init();
        
        console.log(`Platform: ${Clustering.platform}`);
        console.log(`Features:`, Clustering.features);
        
        // Check actual backend being used
        const tf = require('../dist/tf-adapter.js').default;
        console.log(`TensorFlow backend: ${tf.getBackend()}`);
        console.log();
        
        // Generate larger dataset for GPU testing
        console.log('Generating large dataset...');
        const nSamples = 10000;
        const nFeatures = 50;
        const data = [];
        
        // Generate 5 clusters
        for (let cluster = 0; cluster < 5; cluster++) {
            const centerX = cluster * 20;
            const centerY = cluster * 20;
            
            for (let i = 0; i < nSamples / 5; i++) {
                const point = [];
                for (let j = 0; j < nFeatures; j++) {
                    point.push(
                        (j % 2 === 0 ? centerX : centerY) + 
                        Math.random() * 5 - 2.5
                    );
                }
                data.push(point);
            }
        }
        
        console.log(`Dataset: ${data.length} samples, ${nFeatures} features`);
        console.log();
        
        // Benchmark K-Means
        console.log('Benchmarking K-Means...');
        const kmeansLabels = await benchmark('K-Means (5 clusters)', async () => {
            const kmeans = new Clustering.KMeans({ 
                nClusters: 5, 
                nInit: 3,
                maxIter: 100 
            });
            return await kmeans.fitPredict(data);
        });
        
        // Count clusters
        const kmeansCounts = {};
        kmeansLabels.forEach(label => {
            kmeansCounts[label] = (kmeansCounts[label] || 0) + 1;
        });
        console.log('Cluster sizes:', kmeansCounts);
        console.log();
        
        // Benchmark Agglomerative (smaller sample for memory)
        const smallData = data.slice(0, 1000);
        console.log(`Benchmarking Agglomerative (${smallData.length} samples)...`);
        const aggLabels = await benchmark('Agglomerative (5 clusters)', async () => {
            const agg = new Clustering.AgglomerativeClustering({ 
                nClusters: 5,
                linkage: 'ward'
            });
            return await agg.fitPredict(smallData);
        });
        
        // Count clusters
        const aggCounts = {};
        aggLabels.forEach(label => {
            aggCounts[label] = (aggCounts[label] || 0) + 1;
        });
        console.log('Cluster sizes:', aggCounts);
        console.log();
        
        // Memory usage
        const usage = process.memoryUsage();
        console.log('Memory usage:');
        console.log(`  RSS: ${(usage.rss / 1024 / 1024).toFixed(2)} MB`);
        console.log(`  Heap Used: ${(usage.heapUsed / 1024 / 1024).toFixed(2)} MB`);
        console.log(`  External: ${(usage.external / 1024 / 1024).toFixed(2)} MB`);
        
        console.log('\n✅ GPU example completed!');
        console.log('\nTip: For best GPU performance, ensure:');
        console.log('  1. CUDA and cuDNN are properly installed');
        console.log('  2. @tensorflow/tfjs-node-gpu is installed');
        console.log('  3. Your GPU has sufficient memory for the dataset');
        
    } catch (error) {
        console.error('❌ Error:', error.message);
        console.error(error.stack);
        process.exit(1);
    }
}

// Run the example
main();