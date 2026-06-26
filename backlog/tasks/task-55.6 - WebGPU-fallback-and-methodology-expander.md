---
id: TASK-55.6
title: WebGPU feature-detection, graceful fallback, and methodology expander
status: To Do
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

- [ ] #1 On a non-WebGPU browser, the GPU lane gracefully becomes WebGL with a clear banner and no broken UI
- [ ] #2 The methodology expander lists float32, warmup/run counts, the single-readback rule, median reporting, and live config values
- [ ] #3 A permanent footer states the times are from the visitor's own hardware
- [ ] #4 A recorded race GIF/MP4 is shown to visitors whose browser lacks WebGPU
- [ ] #5 ESLint passes before commit

<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->

WebGPU availability as of 2026: Chrome/Edge 113+, Safari (macOS/iOS 26), Firefox 141+ Windows / 145+ macOS-ARM — Firefox Linux/Android not yet shipped. Treat "WebGPU is Baseline" as optimistic shorthand; keep the fallback framing prominent (caniuse ≈82–87% desktop, ≈71% mobile).

<!-- SECTION:NOTES:END -->
