/**
 * Type definitions for clustering-tfjs
 *
 * This file ensures proper type exports for all build targets.
 */

// Re-export everything from the main module
export * from '../src/index';

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

declare module 'clustering-tfjs/utils' {
  export * from '../src/utils/index';
}

declare module 'clustering-tfjs/validation' {
  export * from '../src/validation/index';
}
