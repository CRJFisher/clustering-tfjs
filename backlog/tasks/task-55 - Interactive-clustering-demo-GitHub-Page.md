---
id: TASK-55
title: Interactive clustering demo GitHub Page (WebGPU vs CPU race)
status: To Do
assignee: []
created_date: '2026-06-26'
updated_date: '2026-06-26'
labels:
  - marketing
  - demo
  - webgpu
  - github-pages
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

Build a polished, real-time interactive clustering visualization hosted as a GitHub Page, to serve as the library's primary marketing asset. The page drives GitHub stars and npm installs by making two claims tangible at once:

- **Speed** — a live **WebGPU-vs-CPU race**: the same seeded dataset runs on two backends side by side, each in its own Web Worker, with live wall-clock timers, racing bars, and a "GPU is N.Nx faster" headline. The headline workload is **Spectral RBF affinity construction** (`O(n²·d)`), where the GPU win is real and large (measured ~26× at 10000×100 native vs CPU; ~21× at 1000×50).
- **Trust** — the iconic scikit-learn toy-dataset grid (moons / circles / blobs / anisotropic / no-structure) recreated **live** across all five algorithms (K-Means, Spectral, Agglomerative, HDBSCAN, SOM), signaling parity to every data scientist who already recognizes that image.

One-sentence hook used everywhere (README H1, og:title, Show HN title): **"scikit-learn clustering, GPU-accelerated, 100% in your browser — no Python, no install."**

**Headline feature is a feature gap.** WebGPU is not wired into the library today (only `cpu`/`webgl`/`wasm`/`tensorflow`). Adding `@tensorflow/tfjs-backend-webgpu` behind a per-backend ESM loader is both the one library change worth making and the centerpiece of the launch.

**Load-bearing architecture constraint:** TensorFlow.js maintains a single global engine/backend per JS context (`src/backend/backend.ts` holds a module-level singleton; `Clustering.init` calls `tf.setBackend` once). WebGPU and CPU therefore **cannot** run concurrently on the main thread — a fair race **requires** one dedicated Web Worker per backend, each importing `tfjs-core` + exactly one backend package + the `clustering-tfjs` ESM classes.

**Honesty is a feature, not a footnote.** The demo will face Hacker News scrutiny. Every lane runs float32 with identical input tensors; timing brackets the full `fit_predict` call **including the awaited readback**; first-run shader-compile cost is shown separately and never in the headline multiplier; the small-`n` CPU win is shown as a first-class part of the crossover slider, not buried. A permanent footer states the numbers come from the visitor's own hardware.

**Scope discipline (YAGNI):** ship ONE page — race + grid + code panel + CTA. WASM multi-threading is descoped from v1 (GitHub Pages cannot serve the COOP/COEP headers SharedArrayBuffer needs; the `gzuidhof/coi-serviceworker` shim is the documented workaround but adds a forced first-load reload). WASM/WebGL appear only as optional "also supported" lanes. The existing `examples/` and `examples/observable/` demos are first-gen scaffolding (task-30), not the deploy artifact.

### Delivery milestones

- **M0 (task-55.1)** — Wire up WebGPU + minimal per-backend ESM loader (the one library change).
- **M1 (task-55.2)** — Vite vanilla-TS site skeleton + GitHub Pages deploy pipeline (a blank-but-live page proves the pipeline).
- **M2 (task-55.3, 55.4)** — Race MVP: worker-per-backend harness with the fairness protocol, then the dual-panel race UI. The screenshot-worthy core.
- **M3 (task-55.5, 55.6)** — Crossover slider, methodology expander, graceful WebGPU fallback + recorded race GIF for non-WebGPU/mobile.
- **M4 (task-55.7, 55.8)** — The familiar scikit-learn grid + per-algorithm sliders.
- **M5 (task-55.9)** — Conversion surfaces: live code panel, install CTA, star button, shareable permalinks.
- **M6 (task-55.10)** — Launch polish: README hero, npm metadata, og:image, race GIF/MP4, Show HN / Reddit / Bluesky kit.

See `backlog/docs/interactive-clustering-demo-design.md` for the full design, architecture diagrams, and the fairness protocol.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [ ] #1 A live GitHub Page at `https://CRJFisher.github.io/clustering-tfjs/` shows a working WebGPU-vs-CPU Spectral-affinity race with live timers, racing bars, and an "N.Nx faster" headline, deployed via GitHub Actions on push to main
- [ ] #2 The race is fair and honest: every lane runs float32 on identical input, timing includes the awaited readback, first-run shader-compile cost is shown separately, an always-visible methodology note documents the protocol, and a cross-backend "same result" check is displayed — an equality check on the timed affinity output (the headline workload produces an affinity matrix, not labels, and the async-only WebGPU backend cannot survive `fit_predict`'s sync readback)
- [ ] #3 A crossover `n`-slider (200–5000, hard-capped) visibly flips from CPU-wins at small `n` to GPU-wins past the marked crossover point
- [ ] #4 The scikit-learn toy-dataset grid clusters live across all five algorithms (K-Means, Spectral, Agglomerative, HDBSCAN, SOM) with per-algorithm parameter sliders, on curated datasets/params where float32 parity holds (differences annotated, not hidden)
- [ ] #5 WebGPU is supported in the library via a per-backend ESM loader with `navigator.gpu` feature-detection, `getBackend()` verification, and graceful WebGL→WASM→CPU fallback; non-WebGPU/mobile visitors see a recorded race GIF/MP4
- [ ] #6 Conversion surfaces ship: a code panel mirroring the selected algorithm/backend, an `npm install` one-liner, a persistent GitHub star button, and shareable permalinks encoding dataset+params+n
- [ ] #7 Launch assets ship: README hero (hook → race GIF → live-demo CTA → quickstart), npm description/keywords updated to include hdbscan/som/webgpu/gpu-acceleration, a 1200×630 og:image, a ≤6s looping race GIF/MP4, and a Show HN/Reddit/Bluesky launch kit
- [ ] #8 ESLint passes; the site bundle build is guarded in CI so the live-demo link never rots

<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->

### Investigation summary (multi-agent workflow, 2026-06-26)

A research workflow (8 agents: marketing / UX / tech-architecture / backend-race → adversarial fact-check → synthesis) produced this plan. The fact-checkers returned `minor-issues` on every dimension; corrections folded into the plan:

- The COOP/COEP service-worker shim is **`gzuidhof/coi-serviceworker`** (Guido Zuidhof), not "Tomayac" — Thomas Steiner only blogged about it.
- WebGPU is **not** cleanly "Baseline 2026": Chrome/Edge 113+, Safari (macOS/iOS 26), Firefox 141+ Windows / 145+ macOS-ARM only — **Firefox on Linux/Android has no WebGPU yet**. The graceful-degradation banner is load-bearing, not optional.
- `@tensorflow/tfjs-backend-webgpu` latest is **4.22.0** (~1 year old). Pin `tfjs-core`, `-backend-cpu`, `-backend-webgl`, `-backend-webgpu`, `-backend-wasm` all to the same 4.22.x to avoid kernel-registry mismatches; this locks the demo to the 4.x line (no 5.x backend line is published).
- The "~3× WebGPU-over-WebGL" figure is real but the often-cited TF blog URL 404s; cite `developer.chrome.com/blog/webgpu-io2023` or the tfjs-backend-webgpu README instead. The WebGPU win for **clustering** is workload-dependent — defend it with the repo's own Spectral-affinity numbers, not a generic inference figure.
- **KMeans is the wrong default for the speedup multiplier**: its Lloyd loop reads back every iteration (`await ...data()`) and reduces centroids in a pure-JS double loop, yielding only ~1.2× — use it as animated convergence eye-candy in the grid only. **Spectral RBF affinity** is the headline workload.
- Do **not** reuse `benchmarks/browser_backend.ts` for marketing numbers — it runs a simulated fake kmeans on random tensors. The harness must call the real published library.

### Open questions to resolve during delivery

- KMeans-on-device refactor (one-hot matMul / `unsortedSegmentSum` for the centroid update) so it can race honestly — or accept animation-only for v1? Affects scope of task-55.1.
- Per-backend ESM loader map: contribute into `src/backend/` as a library feature, or keep local to `/site` for v1 (YAGNI, smaller public surface)?
- Exact dataset+param combinations where HDBSCAN/Spectral float32 labels still match the sklearn-parity story (given known probability drift) — curate vs annotate.
- Mobile WebGPU reliability (iOS 26 is new): tested-safe max `n` before OOM; lower the slider cap on detected mobile?
- Permalink schema: how much state to encode, and whether to version it so shared links survive future demo changes.

<!-- SECTION:NOTES:END -->
