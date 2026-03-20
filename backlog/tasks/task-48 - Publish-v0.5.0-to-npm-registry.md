---
id: task-48
title: Publish v0.5.0 to npm registry
status: To Do
assignee: []
created_date: '2026-03-20'
labels: []
dependencies:
  - task-36
---

## Description

clustering-tfjs v0.4.0 is not available on the public npm registry. Consumers cannot npm install it. This task covers the actual publish after task-36 (fix build output) is complete. Should also include verifying the published package is installable and that types resolve correctly.

## Acceptance Criteria

- [ ] Package published to npm as clustering-tfjs with correct version
- [ ] npm install clustering-tfjs resolves and installs successfully
- [ ] TypeScript types resolve correctly in a fresh consumer project
- [ ] All dist files present (index.js and index.esm.js and index.d.ts)
- [ ] CHANGELOG.md updated before publish
