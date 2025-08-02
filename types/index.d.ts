/**
 * Type definitions for clustering-tfjs
 * 
 * This file ensures proper type exports for all build targets.
 */

// Re-export everything from the main module
export * from '../src/index';
export * from '../src/clustering-types';

// Ensure the Clustering namespace is properly typed
import { Clustering as ClusteringImpl } from '../src/clustering';
import type { ClusteringNamespace, DetectedPlatform } from '../src/clustering-types';

declare const Clustering: ClusteringNamespace<DetectedPlatform> & typeof ClusteringImpl;
export { Clustering };

// Module declaration for different environments
declare module 'clustering-tfjs' {
  export * from '../src/index';
}

declare module 'clustering-tfjs/browser' {
  export * from '../src/index';
}

declare module 'clustering-tfjs/node' {
  export * from '../src/index';
}