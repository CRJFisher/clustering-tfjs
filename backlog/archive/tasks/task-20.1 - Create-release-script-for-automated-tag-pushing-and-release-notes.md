---
id: task-20.1
title: Create release script for automated tag pushing and release notes
status: Done
assignee:
  - '@chuck'
created_date: '2025-07-30'
updated_date: '2025-07-30'
labels: []
dependencies: []
parent_task_id: task-20
---

## Description

Create a script that handles the release process: pushes tags, waits for GitHub Actions to complete npm publishing, then updates the GitHub release with provided description

## Acceptance Criteria

- [x] Release script created that accepts version and description
- [x] Script pushes tag and monitors GitHub Actions
- [x] Script updates release notes after successful publish
- [x] Script handles both string and file inputs for description
- [x] Error handling for failed workflows

## Implementation Notes

Created `scripts/release.js` that automates the entire release process:

1. **Features implemented**:
   - Validates version format (v0.1.0 or 0.1.0)
   - Ensures clean working directory on main branch
   - Creates and pushes git tag
   - Monitors GitHub Actions workflow by polling API
   - Creates or updates GitHub release with notes
   - Handles both inline descriptions and file input

2. **Usage patterns**:
   ```bash
   node scripts/release.js v0.1.0 "Release description"
   node scripts/release.js v0.1.0 --file RELEASE_NOTES.md
   ```

3. **Error handling**:
   - Validates prerequisites (gh CLI, main branch, clean directory)
   - Continues with release creation even if workflow monitoring fails
   - Provides clear error messages and next steps

4. **Technical decisions**:
   - Uses temp files for release notes to avoid shell escaping issues
   - Polls GitHub API every 5-10 seconds for workflow status
   - Attempts to edit existing releases if creation fails

5. **npm scripts integration**:
   - Updated package.json with convenience scripts
   - `npm run release` for direct script usage
   - `npm run release:patch/minor/major` for version bumping + release
