/* ------------------------------------------------------------------------- */
/*                           Public Type Exports                             */
/* ------------------------------------------------------------------------- */

export * from './clustering/types';
export * from './types/platform';

// Export the main Clustering namespace for initialization
export { Clustering } from './clustering';
export type { BackendConfig } from './tf-backend';
export type { ClusteringNamespace } from './clustering-types';

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

// SOM visualization utilities
export {
  getComponentPlanes,
  getHitMap,
  getActivationMap,
  trackBMUTrajectory,
  getQuantizationQualityMap,
  getDensityMap,
  getNeighborDistanceMatrix,
  exportForVisualization,
} from './utils/som_visualization';
