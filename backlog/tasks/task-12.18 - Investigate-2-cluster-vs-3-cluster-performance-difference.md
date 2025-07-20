---
id: task-12.18
title: Investigate 2-cluster vs 3-cluster performance difference
status: To Do
assignee: []
created_date: '2025-07-20'
labels: []
dependencies: []
parent_task_id: task-12
---

## Description

Test results show a clear pattern: 3-cluster cases generally perform better than 2-cluster cases. For blobs: n=3 passes perfectly (ARI=1.0) while n=2 fails catastrophically (ARI=0.088). For circles: neither passes but n=3 performs better. This suggests a systematic issue with binary clustering in our implementation.

## Acceptance Criteria

- [ ] Compare spectral embedding dimensions for n=2 vs n=3
- [ ] Check if sklearn has special handling for binary clustering
- [ ] Verify eigenvector selection and k-means behavior for 2D embeddings
- [ ] Document why 2-cluster cases are more challenging
