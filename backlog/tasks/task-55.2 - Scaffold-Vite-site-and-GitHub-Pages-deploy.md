---
id: TASK-55.2
title: Scaffold Vite vanilla-TS site and GitHub Pages deploy pipeline
status: To Do
assignee: []
created_date: '2026-06-26'
updated_date: '2026-06-26'
labels:
  - demo
  - github-pages
  - build
dependencies: []
parent_task_id: TASK-55
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

Create a new `/site` Vite + vanilla-TypeScript single-page app (snake_case source), independent of the library `dist/`, with `base: '/clustering-tfjs/'` (project Pages serve under the repo sub-path, not root). No React/Svelte — a handful of controls plus canvases do not justify a framework's bundle weight; the page must never compete with the GPU for frames.

Add a GitHub Actions workflow on push to main: `actions/configure-pages` → Vite build → `actions/upload-pages-artifact` → `actions/deploy-pages`. Set the repo Pages source to **GitHub Actions** (not a `gh-pages` branch or `docs/` folder, which split source from build and go stale). Single page means no SPA `404.html` trick is needed. Add a CI build guard so the demo link never rots.

Ship a minimal live placeholder page to prove the pipeline end to end before any compute lands.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [ ] #1 A placeholder page is reachable at `https://CRJFisher.github.io/clustering-tfjs/` after a push to main
- [ ] #2 Vite `base` is `/clustering-tfjs/` and no asset or worker URL 404s on the deployed sub-path
- [ ] #3 The GitHub Actions workflow uses `actions/deploy-pages` (no `gh-pages` branch, no `docs/` folder)
- [ ] #4 CI fails if the site bundle fails to build
- [ ] #5 ESLint passes before commit

<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->

Worker URLs must use `import.meta.url` (base-safe). If a wasm binary is added later, `setWasmPaths` must point at the base-prefixed asset URL or it 404s only after deploy. The repo slug is lowercase `clustering-tfjs`.

<!-- SECTION:NOTES:END -->
