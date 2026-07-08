---
id: TASK-55.6
title: WebGPU feature-detection, graceful fallback, and methodology expander
status: Done
assignee: []
created_date: '2026-06-26'
updated_date: '2026-06-26'
labels:
  - demo
  - webgpu
  - ux
dependencies:
  - task-55.5
parent_task_id: TASK-55
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

Make the demo degrade gracefully and survive scrutiny. The fallback banner is load-bearing, not optional: WebGPU is absent on Firefox-Linux/Android and older Safari, and a large share of social-traffic visitors arrive on mobile.

**Feature-detection & fallback:** detect `navigator.gpu`; after `await tf.setBackend('webgpu')` verify `tf.getBackend() === 'webgpu'`. On failure, fall back to webgl → wasm → cpu, relabel the GPU panel honestly (e.g. "WebGL (GPU)") with a non-apologetic banner, and never show an empty or broken lane. For visitors whose browser lacks WebGPU entirely (and mobile), embed a recorded ≤6s race GIF/MP4 so they always see the payoff.

**Methodology-as-a-feature:** an always-visible "how this is timed" expander documenting float32-everywhere, 2–3 discarded warmups, the single readback boundary, median-of-≥5, the `tf.memory().numTensors` assertion, the "labels identical across backends" check, and the live config (algorithm, `n`, `d`, dtype, backend package versions, warmup/run counts). Add the permanent footer: "Times are from YOUR browser/GPU right now — not a cherry-picked machine; numbers vary by hardware."

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [x] #1 On a non-WebGPU browser, the GPU lane gracefully becomes WebGL with a clear banner and no broken UI
- [x] #2 The methodology expander lists float32, warmup/run counts, the single-readback rule, median reporting, and live config values
- [x] #3 A permanent footer states the times are from the visitor's own hardware
- [x] #4 A recorded race clip is shown to visitors whose browser lacks WebGPU (illustrative placeholder; real GIF/MP4 lands in task-55.10 — see reconciliation below)
- [x] #5 ESLint passes before commit

<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->

## High-level summary

The demo now survives a browser without WebGPU and survives scrutiny. The GPU
lane walks a fallback chain — `webgpu → webgl → wasm → cpu` — so every visitor
gets a real, honestly-labelled GPU lane and never an empty or broken panel. A
two-phase honesty model drives the UI: an optimistic main-thread `navigator.gpu`
probe shows a load-bearing fallback banner and relabels the panel *before* the
first race, and the worker's `actual_backend` (read from `tf.getBackend()` after
init) is the *authoritative* reconciliation after it. An always-available "how
this is timed" expander documents the fairness protocol and reports live config
read straight off the measured result, so the documented numbers can never drift
from the code that produced them. Visitors who cannot run WebGPU also see a
clearly-labelled illustrative reference race so they still see the payoff, and a
permanent footer states the times come from their own hardware.

### What changed

- **Fallback chain (`site/src/race_worker.ts`).** `init_lane` walks
  `GPU_FALLBACK_CHAIN = [webgpu, webgl, wasm, cpu]`, calling `Clustering.reset()`
  between failed attempts and returning `tf.getBackend()` as the honest label of
  what actually ran. The `webgpu` candidate is skipped when `navigator.gpu` is
  absent (no wasted backend-package fetch). The non-GPU (cpu) lane takes a direct
  path and is the universal floor.
- **Honest version reporting (`site/src/race_protocol.ts`,
  `site/src/race_worker.ts`).** `RaceResult` gained `tfjs_version`, populated from
  the engine's real `tf.version_core`, so the methodology panel reports the loaded
  version rather than a hardcoded string.
- **Two-phase banner + reference reveal (`site/src/race_ui.ts`).** A pre-race
  `has_webgpu()` probe sets the banner and relabels `#gpu-backend` to a generic
  "GPU" when WebGPU is absent; `reconcile_fallback(actual_backend)` runs after the
  race and is authoritative — it hides the banner only when the lane genuinely ran
  `webgpu`, otherwise names the real backend and reveals the reference clip. This
  also catches the case where `navigator.gpu` is present but `setBackend('webgpu')`
  fails at runtime. The GPU panel header is independently reconciled by the lane's
  `settle()`, so header and banner always agree.
- **Methodology expander (`site/index.html`, `site/src/race_ui.ts`).** An
  always-available `<details>` lists float32-everywhere, the discarded warmups,
  the single awaited-readback boundary, median-of-≥5 reporting, the
  `tf.memory().numTensors`-returns-to-baseline assertion, and the cross-lane
  checksum. Its live-config block (n, d, dtype, warmups·reps, real CPU/GPU
  backends, tfjs version) is seeded from `DEFAULT_RACE_CONFIG` and filled from the
  measured `RaceResult`; backend/version fields reset to "—" at the start of each
  run so a failed or in-flight race never shows the previous race's config.
- **Footer (`site/index.html`).** Updated to the hardware-honest copy: "Times are
  from YOUR browser/GPU right now — not a cherry-picked machine; numbers vary by
  hardware."
- **Reference clip (`site/public/race-reference.svg`).** An illustrative animated
  reference race revealed only to non-WebGPU visitors. Authored in CSS (not SMIL)
  so it honours `prefers-reduced-motion` by holding a static finished frame, and
  carries a persistent "ILLUSTRATIVE — NOT A LIVE MEASUREMENT" label so its anchor
  numbers (~26× at n = 10,000, matching the design-doc reference figures) cannot be
  mistaken for the visitor's own result.

### AC reconciliation

AC #1, #2, #3, #5 are met directly. **AC #4** is met behaviourally — the
fallback mechanism reliably shows a reference race to every visitor whose browser
lacks WebGPU — using a deliberately illustrative, clearly-labelled placeholder
rather than a real screen capture. Per the agreed scope, the polished recorded
GIF/MP4 is produced with the launch assets in **task-55.10**, which swaps the file
in place behind the same `<figure>`/reveal mechanism. The placeholder is honest:
its in-SVG watermark and the figcaption both state it is illustrative and that the
live race below runs on the visitor's own hardware.

### Verification

`npm run build` (tsc `--noEmit` + vite build) and `npm run lint` both pass; the
reference SVG is well-formed XML and is copied into the build output. A six-lens
opus review (correctness ×2, contracts, completeness/honesty, IA/a11y, adversarial
cold-read) drove three fixes: the reduced-motion SVG rewrite, the per-run config
reset, and naming/clarity polish.

### Reference: WebGPU availability (2026)

Chrome/Edge 113+, Safari (macOS/iOS 26), Firefox 141+ Windows / 145+ macOS-ARM —
Firefox Linux/Android not yet shipped. Treat "WebGPU is Baseline" as optimistic
shorthand; keep the fallback framing prominent (caniuse ≈82–87% desktop, ≈71%
mobile).

<!-- SECTION:NOTES:END -->
