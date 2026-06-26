---
id: TASK-55.8
title: Per-algorithm parameter sliders with plain-English captions
status: To Do
assignee: []
created_date: '2026-06-26'
updated_date: '2026-06-26'
labels:
  - demo
  - ux
dependencies:
  - task-55.7
parent_task_id: TASK-55
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

Add distill-style live sliders driving the grid: `n_clusters`; Spectral affinity/`gamma`; HDBSCAN `min_cluster_size`; Agglomerative `linkage`; SOM grid size — each with a one-line plain-English caption explaining its effect. This makes the page link-worthy as an educational explainer (earning durable inbound links and embeds), not just a benchmark.

Changing a slider re-clusters the affected cells live. Controls map to the real library parameter names so the copy-paste code panel (task-55.9) stays truthful.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [ ] #1 Each listed algorithm exposes its key parameter(s) as a live control with a one-line caption
- [ ] #2 Adjusting a slider re-clusters and re-renders the relevant grid cells in real time
- [ ] #3 Controls map to the real library parameter names and produce correct clusterings
- [ ] #4 ESLint passes before commit

<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:NOTES:END -->
