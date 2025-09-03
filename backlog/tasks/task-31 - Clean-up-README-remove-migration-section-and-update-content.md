---
id: task-31
title: Clean up README - remove migration section and update content
status: To Do
assignee:
  - '@assistant'
created_date: '2025-08-03'
updated_date: '2025-08-03'
labels: []
dependencies: []
---

## Description

Remove the 'Migration from scikit-learn' section from README.md, update the Performance section to point to benchmarks, and remove '(no Python required)' from the Features section to make the documentation more focused and professional.

## Acceptance Criteria

- [ ] Remove 'Migration from scikit-learn' section from README.md
- [ ] Update Performance section to link to benchmark results
- [ ] Remove '(no Python required)' text from Features section
- [ ] Ensure all internal links still work after changes
- [ ] Verify README renders correctly on GitHub

## Implementation Plan

1. Remove the entire 'Migration from scikit-learn' section
2. Update the Performance section to reference benchmark results
3. Remove '(no Python required)' from the Features section
4. Check if any other sections reference the migration guide
5. Update the table of contents if present
6. Ensure all internal anchor links work correctly
7. Preview the README to ensure proper markdown rendering
8. Commit changes with clear message

## Specific Changes

### Features Section
Change:
```
- ✅ Pure TypeScript/JavaScript (no Python required)
```
To:
```
- ✅ Pure TypeScript/JavaScript
```

### Table of Contents
Remove line:
```
9. [Migration from scikit-learn](#migration-from-scikit-learn)
```
And renumber subsequent items.

### Performance Section
Current:
```
See [benchmarks/](benchmarks/) for detailed performance data.
```
Should verify this path exists or update to correct location (likely src/benchmarks/).

### Migration Section
Remove entire section starting from `## Migration from scikit-learn` including:
- Python to TypeScript comparison
- Scikit-learn Compatibility subsection
- All content until the next ## heading (Contributing)
