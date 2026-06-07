---
id: DRAFT-2
title: Research backend configuration and testing patterns
status: To Do
assignee: []
created_date: '2025-09-03 21:37'
updated_date: '2025-09-03 21:43'
labels: []
dependencies: []
parent_task_id: TASK-32
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

Research and document the current backend configuration system in clustering-js to understand how backends are currently selected, configured, tested in CI, and documented.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [x] #1 Documented where backend runtime configuration happens
- [x] #2 Identified all backend-related test files and CI workflows
- [x] #3 Found all backend documentation references
- [x] #4 Created list of files that need modification for RN support
- [x] #5 Identified patterns for backend fallback behavior
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->

1. Analyze tf-backend.ts to understand current backend loading mechanism
2. Review existing loader modules (tf-loader.browser.ts, tf-loader.node.ts)
3. Examine test files for backend-specific testing patterns
4. Check CI workflows (.github/workflows/) for backend testing
5. Document all backend-related configuration points
6. List all files requiring modification for RN support
7. Identify fallback patterns used for other backends
8. Review package.json for backend-related dependencies
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->

### Research Findings

#### Current Backend Architecture

- Backend management centralized in src/tf-backend.ts
- Singleton pattern with lazy initialization
- Platform detection based on window and process objects
- Separate loader modules: tf-loader.browser.ts and tf-loader.node.ts

#### Backend Loading Pattern

1. Detection: Check for Node.js vs Browser environment
2. Dynamic import of appropriate loader module
3. Fallback chain: GPU → CPU → Pure JS
4. Backend initialized via tf.setBackend() and tf.ready()

#### Test Infrastructure

- Backend matrix testing in .github/workflows/backend-matrix.yml
- Tests CPU, WebGL, and WASM backends in browser
- Puppeteer-based browser testing with Express server
- No specific mobile/React Native tests currently

#### Type System

- BackendConfig interface in src/tf-backend.ts
- No Platform type enum currently defined
- Backend options: 'cpu', 'webgl', 'wasm', 'node', 'node-gpu'
- Types in src/clustering/types.ts for algorithm params

#### Documentation

- README covers backend selection
- Auto-detection mentioned as primary approach
- Manual backend selection documented
- No React Native documentation

#### Files Requiring Modification for RN

1. src/tf-backend.ts - Add RN detection logic
2. src/tf-loader.rn.ts - New loader module (to create)
3. src/clustering/types.ts - Add Platform type if needed
4. README.md - Add RN section
5. .github/workflows/ - Add RN workflow
6. package.json - Add RN dependencies as optional
<!-- SECTION:NOTES:END -->
