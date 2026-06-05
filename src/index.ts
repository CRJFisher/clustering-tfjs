/* ------------------------------------------------------------------------- */
/*                           Public Type Exports                             */
/* ------------------------------------------------------------------------- */

export * from './clustering/types';
export * from './backend/platform_types';

// Export the main Clustering namespace for initialization
export { Clustering } from './clustering/init';
export type { BackendConfig } from './backend/backend';

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
export { pairwiseDistanceMatrix } from './distance/pairwise_distance';
export { findOptimalClusters } from './model_selection/find_optimal_clusters';
export type {
  ClusterEvaluation,
  FindOptimalClustersOptions,
  OptimalClustersMethod,
} from './model_selection/find_optimal_clusters';
export { computeWss } from './model_selection/compute_wss';
export { findKnee } from './model_selection/kneedle';
export type { KneedleOptions, KneedleResult } from './model_selection/kneedle';

// Validation metrics
export {
  silhouetteSamples,
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
export { adjustedRandIndex } from './validation/adjusted_rand_index';
export {
  normalizedMutualInfo,
  type NMIAverage,
} from './validation/normalized_mutual_info';

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
} from './visualization/som_visualization';
