---
id: task-40
title: Optimize agglomerative clustering from O(n^3) to O(n^2 log n)
status: To Do
assignee: []
created_date: '2026-03-20'
labels: []
dependencies: []
---

## Description

The agglomerative clustering uses a brute-force O(n^3) nearest-pair search with Array.splice() for distance matrix contraction adding O(n^2) per merge step. This makes it impractical for datasets beyond ~500 samples. The implementation should use a min-heap/priority queue for nearest-pair tracking, and for single/complete/average/ward linkage specifically, the NNCHAIN algorithm can achieve O(n^2) amortized complexity. The distance matrix contraction should use index-based active tracking instead of physically removing rows/columns.

## Acceptance Criteria

- [ ] Agglomerative clustering uses priority queue or NNCHAIN for merge step
- [ ] Distance matrix uses index-based active tracking instead of Array.splice
- [ ] 1000-sample dataset completes in under 5 seconds
- [ ] Existing reference tests still pass
- [ ] Benchmark added for 1000 and 5000 sample datasets
