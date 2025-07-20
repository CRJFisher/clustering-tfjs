---
id: task-12.11
title: Fix eigenvector computation to match sklearn
status: To Do
assignee: []
created_date: '2025-07-20'
labels: []
dependencies: []
parent_task_id: task-12
---

## Description

Our eigenvectors have the correct structure (same zero/non-zero pattern) but different numerical values than sklearn's. This is causing the blobs_n2 test to fail with ARI=0.088. sklearn's eigenvectors have exactly 3 unique values per column for disconnected component cases, while ours have many unique values.

## Acceptance Criteria

- [ ] Compare our normalized Laplacian computation with sklearn/scipy line by line
- [ ] Check if eigenvector post-processing or scaling differs
- [ ] Verify our eigenvectors match sklearn's for test cases
- [ ] Achieve same unique-values pattern for disconnected components
- [ ] Fix blobs_n2 to achieve ARI > 0.95
