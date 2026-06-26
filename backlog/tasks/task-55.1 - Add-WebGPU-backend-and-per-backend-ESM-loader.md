---
id: TASK-55.1
title: Add WebGPU backend support and per-backend ESM loader
status: To Do
assignee: []
created_date: '2026-06-26'
updated_date: '2026-06-26'
labels:
  - webgpu
  - backend
  - tfjs
dependencies: []
parent_task_id: TASK-55
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

Add `@tensorflow/tfjs-backend-webgpu` as a selectable backend and introduce a small `backend-name → dynamic-import` map so a caller can request `cpu | webgl | webgpu` and load **only** that backend, not the monolithic `@tensorflow/tfjs` that `src/backend/loader.browser.ts` hard-imports today (which is WebGL-only and the full bundle).

On `'webgpu'`: feature-detect `navigator.gpu`, lazy-`import('@tensorflow/tfjs-backend-webgpu')`, `await tf.setBackend('webgpu')` (resolves to a boolean — `false` if the backend could not initialize), then verify `tf.getBackend() === 'webgpu'` before claiming the lane is real. The failure mode is registration/init failure with fallback to a previously-registered backend, so the `getBackend()` re-check is the correct guard.

Pin `@tensorflow/tfjs-core`, `-backend-cpu`, `-backend-webgl`, `-backend-webgpu` (and `-backend-wasm` if present) to one matching **4.22.x** version to avoid kernel-registry mismatches across lazily-loaded backends. This is the single library change that unblocks the entire race (task-55.3).

This is the one library change worth making; everything else lives under `/site`. Keep Python-style snake_case naming and colocated tests.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [ ] #1 `Clustering.init({ backend: 'webgpu' })` succeeds in a WebGPU-capable browser and `tf.getBackend()` returns `'webgpu'`
- [ ] #2 Requesting a single backend does not bundle the other backend packages (verified via code-split / bundle inspection)
- [ ] #3 `navigator.gpu` absence yields a clean, detectable failure (no uncaught throw) so the caller can fall back to webgl
- [ ] #4 `@tensorflow/tfjs-core`, `-backend-cpu`, `-backend-webgl`, `-backend-webgpu` are pinned to identical 4.22.x versions with no kernel-registry mismatch at runtime
- [ ] #5 A colocated test covers backend selection plus `getBackend()` verification
- [ ] #6 ESLint passes before commit

<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->

Decide as part of this task whether the per-backend loader map lives in `src/backend/` as a contributed library feature or stays local to `/site` for v1 (YAGNI — smaller public surface). The WebGPU backend is async-only (no `dataSync()`); any code path touching it must be fully async.

<!-- SECTION:NOTES:END -->
