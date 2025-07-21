---
id: task-12.22
title: Make k-means less sensitive to small eigenvector differences
status: To Do
assignee: []
created_date: '2025-07-21'
labels: []
dependencies: []
parent_task_id: task-12
---

## Description

The 5 failing RBF tests have eigenvectors that differ from sklearn by up to 0.0065, causing k-means to produce different clusters (ARI ~0.87 instead of 1.0). Investigate ways to make k-means more robust to these small numerical differences.

## Acceptance Criteria

- [ ] K-means produces consistent clusters despite small eigenvector differences
- [ ] Failing RBF tests achieve ARI >= 0.95
- [ ] No regression in passing tests
