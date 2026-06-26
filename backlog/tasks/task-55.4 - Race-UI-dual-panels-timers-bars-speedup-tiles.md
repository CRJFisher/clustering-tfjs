---
id: TASK-55.4
title: 'Race UI: dual scatter panels, live timers, racing bars, speedup tiles'
status: To Do
assignee: []
created_date: '2026-06-26'
updated_date: '2026-06-26'
labels:
  - demo
  - ux
dependencies:
  - task-55.3
parent_task_id: TASK-55
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

Build the hero race UI — the screenshot-worthy core and the launch's minimum viable asset. Two side-by-side panels (left = CPU, right = WebGPU/GPU), each with a lightweight 2D-canvas scatter of the same seeded dataset and, below it, a large live wall-clock timer and a horizontal bar that races left-to-right as the run completes.

Centered headline tiles: (1) ms per backend, (2) the "GPU is N.Nx faster" multiplier (the tweet line), (3) points/sec throughput, (4) a steady-state vs first-run toggle. The first-run (shader-compile) number appears only behind the toggle, never in the headline multiplier.

Display a "same result" indicator backed by an actual cross-backend label-equality check. All scatter rendering stays outside the timed window (no chart redraw inside measured runs) so the page never competes with the measured compute.

The top fold is designed to double as a 1200×630 og:image: fixed dark high-contrast theme, large legible timers/counters, high-contrast cluster colors.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [ ] #1 Both panels render the same seeded dataset and update timers/bars live from worker results
- [ ] #2 The speedup multiplier tile shows the median-based "N.Nx faster" and matches the harness numbers
- [ ] #3 First-run (shader-compile) time is shown only behind the toggle, never in the headline multiplier
- [ ] #4 Scatter rendering occurs outside the timed region (no chart redraw inside measured runs)
- [ ] #5 The "same result" indicator reflects an actual cross-backend label-equality check
- [ ] #6 ESLint passes before commit

<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:NOTES:END -->
