const base = require('./jest.config.js');

/**
 * Coverage gate for the density / decomposition clustering modules
 * (task-50): run via `npm run test:coverage:gate`. See the "Coverage Gate"
 * section of CONTRIBUTING.md for how to read a failure.
 *
 * GATED_MODULES is the single source of truth: each module is instrumented
 * for coverage, held to PER_FILE_THRESHOLD, and its colocated `*.test.ts`
 * suite (derived below) is the test selection for the run. A gated file with
 * no coverage data hard-fails the run ("Coverage data ... not found"), so a
 * renamed module or test surfaces loudly here.
 *
 * `src/clustering/representations.ts` is intentionally absent: it is a
 * type-only module (a single interface) with no runtime code to instrument;
 * its contract is covered behaviourally by the estimator tests in
 * `medoid_selection.test.ts` and `hdbscan.test.ts`.
 */
const GATED_MODULES = [
  'src/clustering/hdbscan.ts',
  'src/clustering/medoid_selection.ts',
  'src/decomposition/pca.ts',
  'src/graph/condensation_tree.ts',
  'src/graph/minimum_spanning_tree.ts',
  'src/graph/mutual_reachability.ts',
];

const PER_FILE_THRESHOLD = {
  branches: 90,
  statements: 95,
  lines: 95,
  functions: 95,
};

module.exports = {
  ...base,
  collectCoverage: true,
  collectCoverageFrom: GATED_MODULES,
  coverageReporters: ['text', 'text-summary'],
  coverageThreshold: Object.fromEntries(
    GATED_MODULES.map((file) => [file, PER_FILE_THRESHOLD]),
  ),
  testMatch: GATED_MODULES.map(
    (file) => `<rootDir>/${file.replace(/\.ts$/, '.test.ts')}`,
  ),
};
