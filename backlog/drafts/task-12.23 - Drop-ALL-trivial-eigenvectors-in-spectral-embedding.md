---
id: task-12.23
title: Drop ALL trivial eigenvectors in spectral embedding
status: To Do
assignee: []
created_date: '2025-07-21'
labels: []
dependencies: []
parent_task_id: task-12
---

## Description

Currently we may only drop the first eigenvector, but sklearn drops ALL eigenvectors with eigenvalues below a threshold (typically 1e-8). This could affect eigenvector selection and final clustering results.

## Acceptance Criteria

- [ ] Identify and drop all eigenvectors with eigenvalues < 1e-8
- [ ] Correct number of meaningful eigenvectors selected
- [ ] Tests show improved clustering accuracy
