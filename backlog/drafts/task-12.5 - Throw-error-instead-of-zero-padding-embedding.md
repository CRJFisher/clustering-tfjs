---
id: task-12.5
title: Throw error instead of zero-padding embedding
status: To Do
assignee: []
created_date: '2025-07-19'
labels: []
dependencies: []
parent_task_id: task-12
---

## Description

When there are fewer than nClusters informative eigenvectors (after removing constant ones), current code may zero-pad. This destroys the pairwise distances in embedding space. Sklearn throws an error in this case. Zero vectors cluster arbitrarily, leading to unstable results.

## Acceptance Criteria

- [ ] Remove any zero-padding logic for embedding matrix
- [ ] Throw descriptive error when informative vectors < nClusters
- [ ] Error message should explain graph connectivity issue
- [ ] Add test case that triggers this error condition
