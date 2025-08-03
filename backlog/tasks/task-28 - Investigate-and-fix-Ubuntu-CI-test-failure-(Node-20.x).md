---
id: task-28
title: Investigate and fix Ubuntu CI test failure (Node 20.x)
status: To Do
assignee:
  - '@assistant'
created_date: '2025-08-03'
updated_date: '2025-08-03'
labels: []
dependencies: []
---

## Description

The main CI test job is failing on Ubuntu with Node 20.x during the 'Run tests' step with exit code 1. This is preventing successful CI completion.

## Acceptance Criteria

- [ ] Root cause of test failure identified and documented
- [ ] Tests pass successfully on Ubuntu with Node 20.x
- [ ] Fix is implemented without breaking other CI jobs
- [ ] CI workflow completes successfully

## Implementation Plan

1. Check GitHub Actions logs for the specific test failure message
2. Reproduce the failure locally with Node 20.x
3. Identify root cause - likely related to tensorflow-helper.ts exports
4. Check for any Node 20.x specific behavior differences
5. Review recent changes to test setup and dependencies
6. Implement fix for the failing tests
7. Verify fix works across all Node versions
8. Update CI configuration if necessary
