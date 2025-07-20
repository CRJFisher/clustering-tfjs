---
id: task-12.6
title: Compare spectral embeddings with sklearn
status: To Do
assignee: []
created_date: '2025-07-20'
labels: []
dependencies: []
parent_task_id: task-12
---

## Description

Since eigensolver accuracy isn't the bottleneck, investigate differences in how the spectral embedding is constructed or used. Compare the actual embedding matrices between our implementation and sklearn's to identify discrepancies.

## Acceptance Criteria

- [ ] Export spectral embeddings from both implementations
- [ ] Compare embeddings element-wise for test cases
- [ ] Identify any normalization or processing differences
- [ ] Fix embedding construction to match sklearn
- [ ] Improve ARI scores on failing tests
