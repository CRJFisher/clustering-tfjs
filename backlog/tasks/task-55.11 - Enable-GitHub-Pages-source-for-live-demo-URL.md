---
id: TASK-55.11
title: Enable GitHub Pages source (GitHub Actions) so the live demo URL resolves
status: Done
assignee: []
created_date: '2026-06-26'
updated_date: '2026-06-28'
labels:
  - github-pages
  - chore
  - ops
dependencies:
  - task-55.2
parent_task_id: TASK-55
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

A one-time manual repository setting that no workflow YAML can apply: set **Settings → Pages → Build and deployment → Source** to **GitHub Actions** for `CRJFisher/clustering-tfjs`. The `deploy-site.yml` workflow (task-55.2) runs `actions/deploy-pages`, which publishes only when the repo's Pages source is GitHub Actions; until this is set, the deploy job errors and `https://CRJFisher.github.io/clustering-tfjs/` does not resolve.

Functionally this depends only on task-55.2's workflow existing on `main`, so it can be done the moment 55.2 merges — it is filed last in the series because it is the final gate that flips the live URL on, not because it must wait for the later content sub-tasks. Doing it early lets every subsequent sub-task (race, grid, code panel, launch assets) verify against the real deployed URL.

This is the step that satisfies AC #1 of task-55.2 (and the live-URL half of the parent task-55 AC #1).

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [x] #1 Repo Settings → Pages → Source is set to "GitHub Actions" (no `gh-pages` branch, no `docs/` folder)
- [x] #2 The "Deploy Site" workflow's `deploy` job completes on a push to main and the `github-pages` environment shows the live URL
- [x] #3 `https://CRJFisher.github.io/clustering-tfjs/` returns 200 and renders the demo page with no asset 404s

<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->

## High-level summary

GitHub Pages for `CRJFisher/clustering-tfjs` is published from GitHub Actions: `Settings → Pages → Source` is set to **GitHub Actions** (`build_type: workflow`), so the `deploy-site.yml` `deploy` job's `actions/deploy-pages` step publishes on every push to `main`. The live demo resolves at `https://crjfisher.github.io/clustering-tfjs/`, the `github-pages` environment records the deployment URL, and the page and its hashed JS/CSS assets all return 200.

This is the final gate of the task-55 series — the one-time repository setting that no workflow YAML can apply. It flips the live URL on now that the whole demo (task-55.1 → task-55.10) is on `main`.

### What was done

- **Enabled the Pages source** via `gh api -X POST repos/CRJFisher/clustering-tfjs/pages -f build_type=workflow`. This is a repository setting, not a committed file; the API confirms `build_type: workflow` and `html_url: https://crjfisher.github.io/clustering-tfjs/`.
- **Landed the task-55 series on `main`.** The whole demo had accumulated on one feature branch rather than merging sub-task by sub-task, so `deploy-site.yml` was never on the default branch and the deploy could not run. The branch was merged to `main` (PR #18, squash), which is what put the workflow on `main` and triggered the first publishing deploy.

### How the acceptance criteria are addressed

- **#1** Pages source is GitHub Actions (`build_type: workflow`) — no `gh-pages` branch and no `docs/` folder.
- **#2** The post-merge `deploy-site.yml` run on `main` completed with `build: success` and `deploy: success`, and the `github-pages` environment shows the live URL.
- **#3** `https://crjfisher.github.io/clustering-tfjs/` returns 200; the bare path 301-redirects to the trailing-slash URL; the entry `index-*.js` and `index-*.css` assets each return 200; no 404s. The page title renders `clustering-tfjs — GPU-accelerated clustering in your browser`. The original AC wording said "placeholder page" from when task-55.2 shipped a placeholder; later sub-tasks replaced it with the full interactive demo, so the deployed page is the demo, not a placeholder.

### Bundle-size regression fixed as a merge prerequisite

Merging the branch surfaced a real regression that CI had never caught (the series never reached `main`, so `main`-gated jobs never ran on it): the `size-check` job failed because `dist/clustering.browser.js` had grown from 128KB to ~1MB. Root cause was in task-55.1's `src/backend/loader.browser.ts`, which dynamic-imports the four tfjs backend packages so a code-splitting bundler emits one lazy chunk each. The UMD library build (`webpack.config.browser.js`) cannot async-load chunks, so webpack inlined all four backends into the single UMD file. The fix externalizes the backend packages alongside the already-external `@tensorflow/tfjs-core`: script-tag/CDN consumers supply a full tfjs as `window.tf` (the loader short-circuits to it and never imports a backend), and the Vite-built demo site consumes the ESM entry where the dynamic imports split correctly. The bundle returned to 129KB, restoring the 150KB budget. Committed as `fix(task-55.1): externalize tfjs backend packages from UMD browser bundle`.

### Operational notes

The setting cannot be committed to the repo; subsequent pushes to `main` redeploy automatically. If a future deploy fails with a "Pages not enabled / configure the Pages source" error, this setting was reverted or never applied — re-run the `gh api` command above.

<!-- SECTION:NOTES:END -->
