---
id: TASK-55.9
title: 'Conversion surfaces: code panel, install CTA, star button, shareable permalinks'
status: Done
assignee: []
created_date: '2026-06-26'
updated_date: '2026-06-27'
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

- [x] #1 The code panel updates to match the selected algorithm/backend and copies valid runnable code
- [x] #2 The install one-liner and a persistent GitHub star button are visible without scrolling away from the hero
- [x] #3 "Share this result" copies a URL that, when opened, restores the same dataset + params + `n`
- [x] #4 All repo/demo links are UTM-tagged for conversion tracking
- [x] #5 ESLint passes before commit

<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->

## High-level summary

The demo turns "cool benchmark" into "I can use this in five minutes." Every cell of the scikit-learn parity grid is now a button: selecting one (mouse or keyboard) drives a **code panel** that shows the real, runnable ~5 lines for that exact cell — the same constructor and parameters the page just ran, on the backend it is actually using — with a Copy button. Because the snippet is generated from the same `resolve_params` the grid fits with, the code can never drift from the rendered result, and live slider tweaks flow straight into it.

The **conversion surfaces** bracket the experience: an `npm install clustering-tfjs` one-liner and a persistent "★ Star on GitHub" button sit in the hero above the fold, and a Section-3 Code + CTA block repeats install/star and adds "Share this result." Every repo link is UTM-tagged through one builder, so no surface can ship an untagged link at runtime.

**Shareable permalinks** are versioned and live in the URL hash (static-Pages-safe, no server, no reload). "Share" encodes the race `n`, the selected cell, and any non-Auto parameter overrides; opening that URL restores all three. The schema carries `v=1`: an unknown version is dropped wholesale rather than guessed, and every field is independently validated — out-of-range numbers clamp to their control's bounds, bogus enums are dropped — so a hand-mangled link can only ever lose state, never break the page.

## What changed

- **Code panel** (`code_panel.ts` + the pure, unit-tested `generate_code.ts`): per-algorithm code generation matching `grid_worker.ts`'s real calls (SOM is the two-call `fit`→`cluster` exception; Spectral emits `gamma` only for rbf and `n_neighbors` only for nearest-neighbors, never an `undefined` literal). The grid backend label narrows to a runnable `Clustering.init({ backend })` arg.
- **Selectable grid cells** (`grid_ui.ts`): each cell is a `role="button"` with `aria-pressed`, a delegated click/keydown handler, arrow-key navigation, and a **roving tabindex** (the grid is one Tab stop). The cell button is the single, live-updated accessible-name carrier; the inner canvas is presentational, so there is no duplicate/nested label.
- **Programmatic restore** (`grid_controls_ui.ts`): `GridControlsPanel.apply_overrides` drives every numeric and select control to a target state through the same paths a user interaction writes, firing exactly one re-cluster. `make_race_ui` and `make_grid_ui` now return small controllers (`RaceController`, `GridSection`) so `main.ts` can orchestrate.
- **Permalink** (`permalink.ts`): a versioned, defensive hash codec (pure `encode_state`/`decode_state` + thin `read`/`write` adapters using `history.replaceState`). **Resolves the delivery open question:** encode `n` + selected cell + non-Auto overrides; version the schema and degrade gracefully on unknown versions / invalid fields.
- **CTA + UTM** (`repo_links.ts`, `clipboard.ts`, `copy_button.ts`, `index.html`, `style.css`): the hero/Section-3 surfaces, a shared copy-with-live-region helper, and one UTM builder. Share has its own live region so a Share right after a Copy does not clobber the announcement.

## Orchestration & restore order

`main.ts` decodes the hash once on load. Selection and race `n` restore immediately (DOM-only / no worker dependency; the race runs exactly once). Override restore is deferred to the grid's `on_backend(live)` callback — only then are the controls enabled and the worker warm — guarded by a latch so a re-fired `on_done` (a later worker crash) cannot re-apply over a dead worker. There is no `hashchange` listener, and `replaceState` writes the share URL without a reload or history entry, so sharing can never retrigger restore. The clipboard is reached only behind the Share/Copy click gesture.

## Review outcome

A multi-lens review (correctness, a11y, spec-completeness, quality, IA) confirmed all five acceptance criteria. Fixes applied: an empty-string permalink value (`n=`) now drops rather than coercing to a bound; the grid cell's duplicate canvas `aria-label`/`role=img` was removed in favour of the button as sole live label; roving tabindex replaced 25 tab stops; Share got a dedicated live region; external links announce "opens in a new tab"; and a shared `get_grid_cell` lookup replaced duplicated cell maps. The static `href`s in the HTML are kept deliberately as a no-JS fallback (JS UTM-tags them before any interaction). Tests: 99 site tests pass (permalink round-trip/degradation, per-algorithm code generation, UTM building); ESLint and the guarded site build are clean.

<!-- SECTION:NOTES:END -->
