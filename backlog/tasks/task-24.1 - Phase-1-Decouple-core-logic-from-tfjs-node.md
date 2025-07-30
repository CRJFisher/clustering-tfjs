---
id: task-24.1
title: 'Phase 1: Decouple core logic from tfjs-node'
status: To Do
assignee: []
created_date: '2025-07-30'
labels: []
dependencies: []
parent_task_id: task-24
---

## Description

Remove all hardcoded tfjs-node imports and create the platform adapter pattern foundation

## Acceptance Criteria

- [ ] All direct imports of @tensorflow/tfjs-node removed
- [ ] tf-adapter.ts module created
- [ ] Core algorithms refactored to use tf-adapter
- [ ] @tensorflow/tfjs-core added to dependencies
- [ ] All tfjs backends moved to peer dependencies
