---
id: TASK-36
title: Fix broken package.json exports and build output structure
status: Done
assignee: []
created_date: '2026-03-20'
updated_date: '2026-03-20 14:16'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The package is currently unpublishable. package.json main/module/types point to dist/index.js, dist/index.esm.js, dist/index.d.ts — none of which exist. The actual build output goes to dist/src/ because tsconfig.json includes test files and has no rootDir. The ESM build script only reads top-level dist-esm/ but output is nested. The release workflow verifies dist/clustering.esm.js which is never produced. Test files are compiled into dist/ and shipped to npm (96 compiled test files). The Platform type has two conflicting definitions exported through different paths. types/index.d.ts creates duplicate/conflicting exports.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 tsconfig sets rootDir to src and excludes test from build compilation
- [x] #2 Build output produces flat dist/index.js and dist/index.esm.js and dist/index.d.ts
- [x] #3 package.json exports map matches actual build output files
- [x] #4 Test files are not compiled into dist/
- [x] #5 Release workflow file verification matches actual build output names
- [x] #6 sideEffects false added to package.json for tree-shaking
- [x] #7 Single Platform type definition used throughout codebase
- [x] #8 types/index.d.ts does not create duplicate or conflicting exports
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create tsconfig.build.json with rootDir:"src" and include:["src"]
2. Update build configs (cjs/esm/types) to extend tsconfig.build.json
3. Delete src/clustering-types.ts and types/index.d.ts (duplicate types)
4. Consolidate Platform type to src/types/platform.ts
5. Simplify package.json exports map and add sideEffects:false
6. Fix release.yml verification file names
7. Fix ESM build scripts to produce complete ESM module tree
8. Fix pre-existing lint errors (replace `as any` with typed Map)
9. Remove dead node-fetch fallback (Node 18+ has global fetch)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
### Approach

The root cause was `tsconfig.json` including `["src", "types", "test"]` with no `rootDir`. TypeScript inferred rootDir as the project root, nesting output under `dist/src/`, `dist/test/`, `dist/types/`. The ESM copy logic used non-recursive `readdirSync` which silently dropped subdirectory modules.

### Changes

**Build configuration:**
- Created `tsconfig.build.json` (extends tsconfig.json, sets `rootDir: "src"`, `include: ["src"]`)
- Updated tsconfig.cjs.json, tsconfig.esm.json, tsconfig.types.json to extend tsconfig.build.json
- Removed "types" from tsconfig.json include (types/ directory deleted)
- Base tsconfig.json kept for IDE/test type-checking (includes src + test)

**ESM build fix:**
- Replaced shallow `readdirSync` copy with complete ESM tree in `dist/esm/`
- `dist/index.esm.js` re-exports from `./esm/index.js` for a self-consistent ESM module graph
- Fixed in both `scripts/build.js` and `scripts/build-multi-platform.js`

**Type deduplication:**
- Deleted `src/clustering-types.ts` (had 4-value Platform type with 'unknown')
- Deleted `types/index.d.ts` (created conflicting re-exports)
- Canonical Platform type: `src/types/platform.ts` (3 values: browser/node/react-native)
- Added PlatformFeatures interface to `src/types/platform.ts`
- Updated `src/clustering.ts` imports to use `./types/platform`

**Package.json:**
- Simplified exports map: `{ types, import, require, default }` (removed nested browser/node conditions)
- Added `"sideEffects": false`

**Workflow:**
- Fixed release.yml verification: `dist/clustering.esm.js` → `dist/index.esm.js`, added `dist/index.d.ts`

**Code quality (bonus):**
- Replaced 16 `(global as any)` casts in PlatformStorage with a typed `Map<string, string>`
- Removed dead node-fetch fallback (engines requires Node 18+ which has global fetch)
<!-- SECTION:NOTES:END -->
