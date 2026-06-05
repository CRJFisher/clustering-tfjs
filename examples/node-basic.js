/**
 * Basic Node.js example for clustering-tfjs
 * 
 * Demonstrates how to use the library in Node.js with automatic backend selection.
 */

const { Clustering } = require('../dist/index.js');

async function main() {
    try {
        console.log('Clustering.js - Node.js Example\n');
        
        // Initialize (optional - will auto-initialize on first use)
        console.log('Initializing clustering library...');
        await Clustering.init();
        
        // Display platform info
        console.log(`Platform: ${Clustering.platform}`);
        console.log(`Features:`, Clustering.features);
        console.log();
        
        // Generate sample data
        const data = [
            // Cluster 1 (bottom-left)
            [1, 2], [1.5, 1.8], [1.1, 2.2], [0.9, 1.9], [1.3, 2.1],
            // Cluster 2 (middle)
            [5, 8], [5.5, 7.8], [5.1, 8.2], [4.9, 7.9], [5.3, 8.1],
            // Cluster 3 (top-right)
            [9, 11], [9.5, 10.8], [9.1, 11.2], [8.9, 10.9], [9.3, 11.1]
        ];
        
        console.log(`Sample data: ${data.length} points`);
        console.log();
        
        // 1. K-Means Clustering
        console.log('1. K-Means Clustering');
        console.log('--------------------');
        const kmeans = new Clustering.KMeans({ n_clusters: 3, n_init: 10 });
        const kmeansLabels = await kmeans.fit_predict(data);
        console.log('Labels:', kmeansLabels);
        console.log();
        
        // 2. Spectral Clustering
        console.log('2. Spectral Clustering');
        console.log('---------------------');
        const spectral = new Clustering.SpectralClustering({ 
            n_clusters: 3,
            affinity: 'rbf',
            gamma: 1.0,
            random_state: 42
        });
        const spectralLabels = await spectral.fit_predict(data);
        console.log('Labels:', spectralLabels);
        console.log();
        
        // 3. Agglomerative Clustering
        console.log('3. Agglomerative Clustering');
        console.log('--------------------------');
        const agglomerative = new Clustering.AgglomerativeClustering({ 
            n_clusters: 3,
            linkage: 'ward'
        });
        const aggLabels = await agglomerative.fit_predict(data);
        console.log('Labels:', aggLabels);
        console.log();
        
        // 4. Self-Organizing Maps (SOM)
        console.log('4. Self-Organizing Maps (SOM)');
        console.log('-----------------------------');
        const som = new Clustering.SOM({
            grid_width: 3,
            grid_height: 3,
            n_clusters: 9,
            topology: 'hexagonal',
            neighborhood: 'gaussian',
            initialization: 'pca',
            num_epochs: 50,
            random_state: 42
        });
        
        const somLabels = await som.fit_predict(data);
        console.log('Labels:', somLabels);
        
        // Get additional SOM information
        const weights = som.getWeights();
        console.log('SOM grid shape:', [weights.length, weights[0].length, weights[0][0].length]); // [3, 3, 2]

        const quantError = som.quantizationError();
        console.log(`Quantization error: ${quantError.toFixed(4)}`);
        console.log();
        
        // 5. Find Optimal Clusters
        console.log('5. Finding Optimal Number of Clusters');
        console.log('------------------------------------');
        const { find_optimal_clusters } = require('../dist/index.js');
        
        const result = await find_optimal_clusters(data, {
            min_clusters: 2,
            max_clusters: 5,
            algorithm: 'kmeans'
        });
        
        console.log(`Optimal number of clusters: ${result.optimal.k}`);
        console.log(`Silhouette score: ${result.optimal.silhouette.toFixed(4)}`);
        console.log('\nAll evaluations:');
        result.evaluations.forEach(eval => {
            console.log(`  k=${eval.k}: silhouette=${eval.silhouette.toFixed(4)}`);
        });
        
        console.log('\n✅ All examples completed successfully!');
        
    } catch (error) {
        console.error('❌ Error:', error.message);
        console.error(error.stack);
        process.exit(1);
    }
}

// Run the example
main();