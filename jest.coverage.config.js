const base = require('./jest.config.js');

/**
 * Coverage gate for the density / decomposition clustering modules
 * (task-50): run via `npm run test:coverage:gate`, which scopes the test run
 * to these modules' colocated suites.
 *
 * Every file listed in `coverageThreshold` must also be in
 * `collectCoverageFrom` AND be instrumented by the executed tests — Jest
 * hard-fails with "Coverage data ... not found" otherwise. Keep this list,
 * `collectCoverageFrom`, and the `--runTestsByPath` arguments of the gate
 * script in lockstep.
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
  'src/distance/kdistance.ts',
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
  coverageReporters: ['text', 'text-summary', 'lcov'],
  coverageThreshold: Object.fromEntries(
    GATED_MODULES.map((file) => [file, PER_FILE_THRESHOLD]),
  ),
};
