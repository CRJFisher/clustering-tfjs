export * from './clustering/types';
export * from './backend/platform_types';

export { Clustering } from './clustering/init';
export type { BackendConfig } from './backend/backend';

export { AgglomerativeClustering } from './clustering/agglomerative';
export {
  SpectralClustering,
  type LaplacianResult,
  type EmbeddingResult,
  type IntermediateSteps,
  type DebugInfo,
} from './clustering/spectral';
export { KMeans, type KMeansJSON } from './clustering/kmeans';
export { HDBSCAN } from './clustering/hdbscan';
export { SOM } from './clustering/som';

export {
  PCA,
  type PCAParams,
  type PCAJSON,
  type EigResult,
  power_iteration_eig,
} from './decomposition/pca';

export type { ClusterRepresentations } from './clustering/representations';
export {
  select_medoids,
  type MedoidResult,
} from './clustering/medoid_selection';

export { pairwise_distance_matrix } from './distance/pairwise_distance';
export { find_optimal_clusters } from './model_selection/find_optimal_clusters';
export type {
  ClusterEvaluation,
  FindOptimalClustersOptions,
  OptimalClustersMethod,
} from './model_selection/find_optimal_clusters';
export { compute_wss } from './model_selection/compute_wss';
export { find_knee } from './model_selection/kneedle';
export type { KneedleOptions, KneedleResult } from './model_selection/kneedle';

export {
  silhouette_samples,
  silhouette_score,
  silhouette_score_subset,
  type ValidationMetric,
} from './validation/silhouette';
export {
  davies_bouldin,
  davies_bouldin_efficient,
} from './validation/davies_bouldin';
export {
  calinski_harabasz,
  calinski_harabasz_efficient,
} from './validation/calinski_harabasz';
export { adjusted_rand_index } from './validation/adjusted_rand_index';
export {
  normalized_mutual_info,
  type NMIAverage,
} from './validation/normalized_mutual_info';

export {
  get_component_planes,
  get_hit_map,
  get_activation_map,
  track_bmu_trajectory,
  get_quantization_quality_map,
  get_density_map,
  get_neighbor_distance_matrix,
  export_for_visualization,
} from './visualization/som_visualization';
