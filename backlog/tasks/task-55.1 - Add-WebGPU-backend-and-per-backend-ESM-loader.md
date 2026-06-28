---
id: TASK-55.1
title: Add WebGPU backend support and per-backend ESM loader
status: Done
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

- [~] #1 `Clustering.init({ backend: 'webgpu' })` succeeds in a WebGPU-capable browser and `tf.getBackend()` returns `'webgpu'` (code-complete + unit-covered; real-browser check lands with /site, task-55.3)
- [~] #2 Requesting a single backend does not bundle the other backend packages (mechanism verified: distinct literal `import()` specifiers, no eager monolith import; bundle inspection lands with /site, task-55.2)
- [x] #3 `navigator.gpu` absence yields a clean, detectable failure (no uncaught throw) so the caller can fall back to webgl
- [x] #4 `@tensorflow/tfjs-core`, `-backend-cpu`, `-backend-webgl`, `-backend-webgpu` are pinned to identical 4.22.x versions with no kernel-registry mismatch at runtime
- [x] #5 A colocated test covers backend selection plus `getBackend()` verification
- [x] #6 ESLint passes before commit

<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->

## High-level summary

WebGPU is the centerpiece of the launch demo's speed claim, but the library could only reach a browser GPU through `webgl` bundled inside the monolithic `@tensorflow/tfjs` build. This work makes the browser backend selectable and lazily loaded: a caller asks for `cpu | webgl | webgpu | wasm` and the library loads `@tensorflow/tfjs-core` plus exactly that one backend package, never the others.

The per-backend loader lives in `src/backend/` as a library feature rather than local to `/site` — the public contract is `Clustering.init({ backend })`, and the same loader is what the race workers (task-55.3) reuse. `loader.browser.ts` holds a `backend-name → dynamic-import()` map keyed by `TensorFlowBackend`; each entry is a distinct static `import()` specifier so a code-splitting bundler can emit one chunk per backend. Because `webgl` and `webgpu` both depend on `-backend-cpu`, the cpu kernels ride along as the fallback whenever a GPU lane is selected; only the webgl-vs-webgpu chunks are mutually exclusive.

The browser path is deliberately self-contained: it owns flag application, `setBackend`, readiness, and — for WebGPU — the honesty protocol. WebGPU initialization feature-detects `navigator.gpu` before importing the heavy package (a clean, catchable failure when it is absent), then after `setBackend('webgpu')` re-checks `getBackend() === 'webgpu'`, because `setBackend` resolves `false` and silently keeps a prior backend when the GPU lane fails to initialize. The node and react-native loaders are untouched; `backend.ts` early-returns the browser loader and keeps the generic flag/setBackend/ready tail for the other platforms.

Start reading at `src/backend/loader.browser.ts` (the map and the WebGPU flow), then `src/backend/backend.ts::load_backend` for the platform fork. `src/backend/loader.browser.test.ts` exercises selection per backend, `navigator.gpu` detection, the `getBackend()` verification (both the `setBackend`-false and silent-fallback cases), and the `window.tf` script-tag hatch.

Watch: the `'webgpu'` backend is async-only — it initializes and verifies, but the clustering algorithms still read tensors back synchronously, so running `fit`/`fit_predict` on `'webgpu'` is not yet supported (documented on `Clustering.init`). The demo's race harness drives WebGPU with raw tensors, not the algorithms. Making the algorithms webgpu-safe end-to-end is a separate follow-up.

### Verification status of acceptance criteria

- **AC #1 / #2** are code-complete and unit-covered but their final empirical check is downstream: a real `getBackend() === 'webgpu'` needs a WebGPU-capable browser, and true chunk-splitting needs the `/site` Vite bundle (task-55.2/55.3). This task guarantees the *mechanism* — distinct literal `import()` specifiers and no eager monolith import, both asserted against the built ESM output.
- **AC #3, #4, #5, #6** are fully satisfied here (clean `navigator.gpu` failure, identical 4.22.0 pins across core/cpu/webgl/webgpu/wasm, colocated test, ESLint clean).

### Review follow-ups noted, not actioned

Surfaced by review, deliberately deferred: README/examples still load `@tensorflow/tfjs@4.20.0` via CDN script tag (version skew vs the 4.22.0 pins) — reconcile with launch assets in task-55.10; `Clustering.features` advertises WebGL but has no `webgpu` capability flag (the catchable init throw is the detection path for now); raw `setBackend`/dynamic-import failures surface tfjs-native messages rather than the loader's curated wording (the failures remain catchable rejections, so the fallback contract holds).

<!-- SECTION:NOTES:END -->
