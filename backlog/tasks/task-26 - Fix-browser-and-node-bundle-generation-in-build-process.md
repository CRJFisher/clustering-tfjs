---
id: task-26
title: Fix browser and node bundle generation in build process
status: To Do
assignee: []
created_date: '2025-08-03'
labels: []
dependencies: []
---

## Description

The build process is only creating dist/index.js but not dist/clustering.browser.js and dist/clustering.node.js, causing size-check and other tests to fail

## Acceptance Criteria

- [ ] dist/clustering.browser.js is created by build process
- [ ] dist/clustering.node.js is created by build process
- [ ] size-check job passes
- [ ] All bundle files have appropriate sizes
