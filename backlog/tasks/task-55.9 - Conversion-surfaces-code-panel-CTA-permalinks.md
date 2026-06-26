---
id: TASK-55.9
title: 'Conversion surfaces: code panel, install CTA, star button, shareable permalinks'
status: To Do
assignee: []
created_date: '2026-06-26'
updated_date: '2026-06-26'
labels:
  - demo
  - marketing
dependencies:
  - task-55.6
  - task-55.8
parent_task_id: TASK-55
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

Turn the wow-demo into installs and shares — the bridge from "cool demo" to "I can use this in 5 minutes" (the Three.js "every demo links to its source" pattern).

- **Code panel** mirroring the currently selected algorithm + backend, showing the real ~5 lines (`await Clustering.init({ backend: 'webgpu' }); new SpectralClustering({ ... }); await model.fit_predict(X)`) with a Copy button.
- **Install one-liner:** `npm install clustering-tfjs`, visible without scrolling away from the hero.
- **Persistent "Star on GitHub" button** in the demo header.
- **Shareable permalinks:** a "Share this result" button plus on-load parsing that encodes/decodes the current dataset + params + `n` in the URL, so anyone can reproduce or tweet a specific result.

All repo/demo links are UTM-tagged for conversion tracking.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [ ] #1 The code panel updates to match the selected algorithm/backend and copies valid runnable code
- [ ] #2 The install one-liner and a persistent GitHub star button are visible without scrolling away from the hero
- [ ] #3 "Share this result" copies a URL that, when opened, restores the same dataset + params + `n`
- [ ] #4 All repo/demo links are UTM-tagged for conversion tracking
- [ ] #5 ESLint passes before commit

<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->

Open question for delivery: how much state to encode in the permalink and whether to version the schema so shared links survive future demo changes.

<!-- SECTION:NOTES:END -->
