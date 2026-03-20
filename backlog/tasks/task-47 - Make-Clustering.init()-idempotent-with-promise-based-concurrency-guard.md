---
id: task-47
title: Make Clustering.init() idempotent with promise-based concurrency guard
status: To Do
assignee: []
created_date: '2026-03-20'
labels: []
dependencies:
  - task-43
---

## Description

If two callers invoke Clustering.init() concurrently (before the first resolves), the behavior is undefined — it could double-initialize the backend, throw, or silently race. Real consumers (code-charter) had to implement their own promise-based singleton wrapper. This should be handled internally. Depends on task-43 (fixing the TF.js backend architecture) being completed first.

## Acceptance Criteria

- [ ] Clustering.init() is internally idempotent — concurrent calls return the same promise
- [ ] Second call to init() with same config is a no-op that resolves immediately
- [ ] Second call with different config either throws a clear error or re-initializes safely
- [ ] Behavior documented in JSDoc
- [ ] Unit test verifies concurrent init() calls resolve correctly
