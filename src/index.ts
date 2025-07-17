/* ------------------------------------------------------------------------- */
/*                           Public Type Exports                             */
/* ------------------------------------------------------------------------- */

export * from "./clustering/types";

// Public estimators
export { AgglomerativeClustering } from "./clustering/agglomerative";
export { SpectralClustering } from "./clustering/spectral";
export { KMeans } from "./clustering/kmeans";

// Utilities
export { pairwiseDistanceMatrix } from "./utils/pairwise_distance";

// Graph Laplacian helpers (task-10)
export {
  degree_vector,
  normalised_laplacian,
  jacobi_eigen_decomposition,
  smallest_eigenvectors,
} from "./utils/laplacian";

// Deterministic eigenpair post-processing
export { deterministic_eigenpair_processing } from "./utils/eigen_post";
