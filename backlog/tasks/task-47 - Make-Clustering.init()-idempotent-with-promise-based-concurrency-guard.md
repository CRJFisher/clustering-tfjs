---
id: task-47
title: Make Clustering.init() idempotent with promise-based concurrency guard
status: Done
assignee:
  - '@claude'
created_date: '2026-03-20'
labels: []
dependencies:
  - task-43
---

## Description

If two callers invoke Clustering.init() concurrently (before the first resolves), the behavior is undefined — it could double-initialize the backend, throw, or silently race. Real consumers (code-charter) had to implement their own promise-based singleton wrapper. This should be handled internally. Depends on task-43 (fixing the TF.js backend architecture) being completed first.

## Acceptance Criteria

- [x] Clustering.init() is internally idempotent — concurrent calls return the same promise
- [x] Second call to init() with same config is a no-op that resolves immediately
- [x] Second call with different config either throws a clear error or re-initializes safely
- [x] Behavior documented in JSDoc
- [x] Unit test verifies concurrent init() calls resolve correctly

## Implementation Plan

1. Add config normalization utilities (`sortedStringify`, `stripEmpty`, `configKey`) to `clustering.ts`
2. Add promise caching (`initPromise`, `initConfigKey`) at the `Clustering.init()` level
3. Change `init()` from `async` to synchronous function returning cached `Promise<void>`
4. Add synchronous throw for config conflicts with clear error message
5. Add `Clustering.reset()` method that clears both layers
6. Simplify `initializeBackend()` by removing the backend-switch path (conflicts with idempotency)
7. Write comprehensive tests for concurrency, config normalization, conflicts, reset, and error recovery

## Implementation Notes

- **Config comparison**: Uses `sortedStringify` (deterministic JSON with sorted keys) + `stripEmpty` (normalizes empty sub-objects and undefined values) to produce a stable config key. Handles edge cases like `{ flags: {} }` == `{}` and `{ flags: { X: undefined } }` == `{}`.
- **Promise caching at Clustering.init() level**: The promise is cached as a module-level variable. Concurrent callers get the exact same `Promise` object (reference equality). `initializeBackend()` retains its own simpler dedup as defense-in-depth.
- **Synchronous throw for config conflicts**: Chosen over re-initialization because switching backends mid-flight can invalidate in-progress computations. Error message directs users to `Clustering.reset()`.
- **Removed backend-switch path from `initializeBackend()`**: The old `if (tfInstance && config.backend)` path that called `setBackend()` on an already-initialized instance conflicted with the idempotency guarantee. Users who need to switch backends now call `Clustering.reset()` then `Clustering.init()`.
- **Modified files**: `src/clustering.ts`, `src/tf-backend.ts`, `test/tf-backend.test.ts`, `test/clustering-init.test.ts` (new)
