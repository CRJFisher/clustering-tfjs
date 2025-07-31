# Scripts

This directory contains utility scripts for development and release management.

## release.js

Automated release script that:
1. Creates and pushes a git tag
2. Monitors GitHub Actions for the release workflow
3. Updates the GitHub release with provided notes after successful npm publish

### Usage

```bash
# With inline description
node scripts/release.js v0.1.0 "Initial release with clustering algorithms"

# With release notes from file
node scripts/release.js v0.1.1 --file RELEASE_NOTES.md

# Using npm scripts
npm run release v0.1.0 "Bug fixes and performance improvements"
```

### Requirements

- GitHub CLI (`gh`) must be installed and authenticated
- Must be run from the main branch with a clean working directory
- Requires appropriate GitHub permissions to create releases

### How it works

1. Validates the version format and checks prerequisites
2. Creates a git tag and pushes it to GitHub
3. Waits for the GitHub Actions release workflow to start
4. Monitors the workflow progress
5. Creates or updates the GitHub release with the provided description

If the workflow fails, the script will exit with an error but the tag will remain pushed.

## build.js

Custom build script that generates CommonJS and ES module outputs for npm distribution.

## Other Scripts

- `benchmark.ts` - Performance benchmarking for clustering algorithms
- `compare-benchmarks.ts` - Compare benchmark results across versions