import type { Config } from "@jest/types";

// Prevent Jest from attempting to write the haste map to the filesystem which
// is disallowed in the execution sandbox.
process.env.JEST_DISABLE_FS_CACHE = "1";

const config: Config.InitialOptions = {
  preset: "ts-jest",
  testEnvironment: "node",
  // Disable any disk caching to avoid EPERM errors in the sandbox
  cache: false,
  roots: ["<rootDir>/test"],
  moduleFileExtensions: ["ts", "js", "json"],
  collectCoverageFrom: ["src/**/*.{ts,tsx}"],
  coverageDirectory: "coverage",
};

export default config;
