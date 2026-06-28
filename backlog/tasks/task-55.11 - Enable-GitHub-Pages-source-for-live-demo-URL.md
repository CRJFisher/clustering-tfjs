---
id: TASK-55.11
title: Enable GitHub Pages source (GitHub Actions) so the live demo URL resolves
status: To Do
assignee: []
created_date: '2026-06-26'
updated_date: '2026-06-26'
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

- [ ] #1 Repo Settings → Pages → Source is set to "GitHub Actions" (no `gh-pages` branch, no `docs/` folder)
- [ ] #2 The "Deploy Site" workflow's `deploy` job completes on a push to main and the `github-pages` environment shows the live URL
- [ ] #3 `https://CRJFisher.github.io/clustering-tfjs/` returns 200 and renders the placeholder page with no asset 404s

<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->

The setting lives under the repo's web UI (or `gh api -X POST repos/CRJFisher/clustering-tfjs/pages -f build_type=workflow`); it cannot be committed to the repo. After enabling, the first qualifying push to `main` triggers the deploy; subsequent pushes redeploy automatically. If the deploy job ever fails with a "Pages not enabled / configure the Pages source" error, this setting was reverted or never applied.

<!-- SECTION:NOTES:END -->
