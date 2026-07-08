---
id: TASK-55.2
title: Scaffold Vite vanilla-TS site and GitHub Pages deploy pipeline
status: Done
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
- [x] #2 Vite `base` is `/clustering-tfjs/` and no asset or worker URL 404s on the deployed sub-path
- [x] #3 The GitHub Actions workflow uses `actions/deploy-pages` (no `gh-pages` branch, no `docs/` folder)
- [x] #4 CI fails if the site bundle fails to build
- [x] #5 ESLint passes before commit

<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->

## High-level summary

A new `/site` folder holds a Vite + vanilla-TypeScript single-page app that builds independently of the library `dist/` and deploys to GitHub Pages. It ships a branded dark-theme placeholder — the launch hook plus a "coming soon" note — whose only job is to prove the deploy pipeline end to end before any compute lands. The race, the sklearn grid, the code panel, and the library dependency all arrive in later sub-tasks; M1 deliberately carries no `clustering-tfjs` import, no worker, and no wasm.

The site sets `base: '/clustering-tfjs/'` so every emitted asset URL carries the project-Pages sub-path prefix — the slug is the lowercase GitHub repo name. A push to `main` (or a manual `workflow_dispatch`) runs the Pages deploy chain; every pull request runs the same bundle build as a guard so the live-demo link never rots.

### What was built

- **`site/`** — `package.json` (own `vite`/`typescript` devDeps, `private`, `type: module`), `vite.config.ts` (just `base`), a single strict `tsconfig.json` covering both `src/` and `vite.config.ts`, `index.html` (the placeholder + inline data-URI favicon, no external assets), `src/main.ts` (imports the stylesheet so Vite rewrites its URL with the base prefix), `src/style.css`, and `src/vite-env.d.ts` (Vite client types for the CSS side-effect import). A committed `site/package-lock.json` lets CI run `npm ci`.
- **`.github/workflows/deploy-site.yml`** — a `build` job (push + pull_request) that runs `npm ci` and `npm run build` in `site/`, and a `deploy` job (`needs: build`, runs on anything that is not a PR) that publishes via `actions/configure-pages` → `actions/upload-pages-artifact` (path `site/dist`) → `actions/deploy-pages`. Pages-scoped least-privilege `permissions`, a `github-pages` environment, and a deploy-scoped `concurrency` group so PR guard builds never queue behind a publish.

### How the acceptance criteria are addressed

- **#2** `base` is `/clustering-tfjs/`; `vite build` emits both the JS and CSS with that prefix and `vite preview` serves `200` at the sub-path (bare `/` `302`-redirects to it). No worker/wasm exists yet to base-break.
- **#3** The workflow uses `actions/deploy-pages`; Pages source is GitHub Actions — no `gh-pages` branch, no `docs/` folder.
- **#4** The build script is `tsc --noEmit && vite build`, so a type error or a bundle failure exits non-zero and fails the workflow; the PR `build` job is the guard.
- **#5** The repo's ESLint (`eslint src test_support benchmarks`) passes; it does not yet cover `site/` (see the deferral note below).

### Deploy prerequisites (why #1 is unchecked)

AC #1 is confirmable only after the work merges to `main` and the repo's **Settings → Pages → Source** is set to **GitHub Actions** (a one-time manual setting no YAML can apply, tracked as task-55.11). Until both happen, the live URL does not resolve; the pipeline itself is verified end to end locally.

### Build guard caught a real issue

TypeScript 6.0 rejects a bare `import "./style.css"` without ambient declarations, so the build initially failed at `tsc`. `src/vite-env.d.ts` (`/// <reference types="vite/client" />`) supplies those types — the standard Vite vanilla-TS fix — confirming the `tsc --noEmit` guard has teeth.

Worker URLs must use `import.meta.url` (base-safe). If a wasm binary is added later, `setWasmPaths` must point at the base-prefixed asset URL or it 404s only after deploy. The repo slug is lowercase `clustering-tfjs`. Note that `site/` is intentionally outside the repo's ESLint scope for M1 (its only TypeScript is a one-line CSS import); wiring `site/` into ESLint with the project's snake_case rules is deferred to task-55.3, where real site logic lands.

<!-- SECTION:NOTES:END -->
