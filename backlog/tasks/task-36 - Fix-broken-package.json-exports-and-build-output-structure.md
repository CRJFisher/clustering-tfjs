---
id: task-36
title: Fix broken package.json exports and build output structure
status: To Do
assignee: []
created_date: '2026-03-20'
labels: []
dependencies: []
---

## Description

The package is currently unpublishable. package.json main/module/types point to dist/index.js, dist/index.esm.js, dist/index.d.ts — none of which exist. The actual build output goes to dist/src/ because tsconfig.json includes test files and has no rootDir. The ESM build script only reads top-level dist-esm/ but output is nested. The release workflow verifies dist/clustering.esm.js which is never produced. Test files are compiled into dist/ and shipped to npm (96 compiled test files). The Platform type has two conflicting definitions exported through different paths. types/index.d.ts creates duplicate/conflicting exports.

## Acceptance Criteria

- [ ] tsconfig sets rootDir to src and excludes test from build compilation
- [ ] Build output produces flat dist/index.js and dist/index.esm.js and dist/index.d.ts
- [ ] package.json exports map matches actual build output files
- [ ] Test files are not compiled into dist/
- [ ] Release workflow file verification matches actual build output names
- [ ] sideEffects false added to package.json for tree-shaking
- [ ] Single Platform type definition used throughout codebase
- [ ] types/index.d.ts does not create duplicate or conflicting exports
