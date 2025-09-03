/* ------------------------------------------------------------------------- */
/*                           Public Type Exports                             */
/* ------------------------------------------------------------------------- */

export * from './clustering/types';
export * from './types/platform';

// Export the main Clustering namespace for initialization
export { Clustering } from './clustering';
export type { BackendConfig } from './tf-backend';

// Public estimators
export { AgglomerativeClustering } from './clustering/agglomerative';
export {
  SpectralClustering,
  type LaplacianResult,
  type EmbeddingResult,
  type IntermediateSteps,
  type DebugInfo,
} from './clustering/spectral';
export { KMeans } from './clustering/kmeans';
export { SOM } from './clustering/som';

// Utilities
export { pairwiseDistanceMatrix } from './utils/pairwise_distance';
export { findOptimalClusters } from './utils/findOptimalClusters';
export type {
  ClusterEvaluation,
  FindOptimalClustersOptions,
} from './utils/findOptimalClusters';

// Validation metrics
export {
  silhouetteScore,
  silhouetteScoreSubset,
} from './validation/silhouette';
export {
  daviesBouldin,
  daviesBouldinEfficient,
} from './validation/davies_bouldin';
export {
  calinskiHarabasz,
  calinskiHarabaszEfficient,
} from './validation/calinski_harabasz';

// Graph Laplacian helpers (task-10)
export {
  degree_vector,
  normalised_laplacian,
  jacobi_eigen_decomposition,
  smallest_eigenvectors,
} from './utils/laplacian';

// Deterministic eigenpair post-processing
export { deterministic_eigenpair_processing } from './utils/eigen_post';

// SOM-specific utilities
export {
  initializeWeights,
  findBMU,
  findBMUBatch,
  gaussianNeighborhood,
  bubbleNeighborhood,
  mexicanHatNeighborhood,
  linearDecay,
  exponentialDecay,
  DecayTracker,
} from './clustering/som_utils';

export {
  getComponentPlanes,
  getHitMap,
  getActivationMap,
  trackBMUTrajectory,
  getQuantizationQualityMap,
  exportForVisualization,
} from './utils/som_visualization';
