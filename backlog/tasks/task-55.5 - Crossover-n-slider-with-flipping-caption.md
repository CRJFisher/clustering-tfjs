---
id: TASK-55.5
title: Crossover n-slider with marked crossover point and flipping caption
status: To Do
assignee: []
created_date: '2026-06-26'
updated_date: '2026-06-26'
labels:
  - demo
  - ux
dependencies:
  - task-55.4
parent_task_id: TASK-55
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

Add the `n`-slider (200 to a hard cap of 5000) that re-runs the race live and flips the caption between "At small n, CPU wins — GPU transfer + dispatch cost more than the math" and "GPU pulls ahead." This is the most shareable, least cherry-pick-able interaction: making the small-`n` CPU win a first-class, visible part of the demo is what defends it against accusations of rigging.

Detect and mark the crossover `n` on the slider track — the point where the two wall-time curves intersect (fixed overhead: GPU dispatch + upload + single readback ≈ a few ms constant, CPU ≈ near zero; compute grows with `n²·d`). The 5000 cap is enforced because the dense `O(n²)` affinity matrix risks OOM/jank on low-end and mobile devices above it; large-`n` runs must not freeze the main thread (compute is in workers).

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [ ] #1 Dragging the slider re-runs the race and updates bars/timers for the new `n`
- [ ] #2 Below the crossover the CPU lane wins and the caption reflects it; above it the GPU wins
- [ ] #3 The crossover `n` is computed and visibly marked on the slider track
- [ ] #4 The slider cannot exceed `n=5000`; large-`n` runs do not freeze the main thread
- [ ] #5 ESLint passes before commit

<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->

Open question for delivery: lower the cap specifically on detected mobile devices (iOS 26 WebGPU is new; tested-safe max `n` before OOM is unknown).

<!-- SECTION:NOTES:END -->
