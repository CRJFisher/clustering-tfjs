---
id: task-27
title: Fix Windows Jest configuration for TensorFlow backend
status: To Do
assignee: []
created_date: '2025-08-03'
labels: []
dependencies: []
---

## Description

The Jest moduleNameMapper redirects tfjs-node to tfjs on Windows CI, but tests fail with 'Backend name tensorflow not found in registry' error

## Acceptance Criteria

- [ ] Windows tests pass without backend errors
- [ ] Tests can use appropriate TensorFlow backends on Windows
- [ ] No regression in test functionality
