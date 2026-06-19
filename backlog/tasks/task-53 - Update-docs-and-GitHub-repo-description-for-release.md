---
id: task-53
title: Update docs and GitHub repo description for release
status: To Do
assignee: []
created_date: '2026-06-15 20:19'
labels: []
dependencies: []
---

## Description

Bring all documentation up to date to reflect the algorithms and API available at release time, including HDBSCAN and the full clustering suite. Update the GitHub repository description and topics to match.

## Acceptance Criteria

<!-- AC:BEGIN -->

- [x] #1 README.md reflects current algorithm list (KMeans, Agglomerative, Spectral, SOM, HDBSCAN) and API,examples/observable/README.md updated if relevant,scripts/README.md and tools/README.md checked and updated,GitHub repo description and topics updated to reflect current scope,No references to TDT or old naming remain in public-facing docs
<!-- AC:END -->

## Implementation Notes

## High-level summary

The library's public documentation is now consistent with the five-algorithm release. README.md previously listed only four algorithms and contained a broken CI badge pointing to an archived `clustering-js` repo, a stale `// clustering-js` code comment, and an "under active development" caveat inappropriate for a release. `tools/README.md` contained camelCase API examples (`nClusters`, `captureDebugInfo`, `getDebugInfo`, `fitWithIntermediateSteps`) reflecting a naming convention the library no longer uses, plus a "Key Findings" section that recorded internal debugging history rather than documenting the tool.

The approach was to bring all four documents to a canonical, present-tense state rather than patching individual lines. HDBSCAN was added consistently across all sections where algorithms appear — features list, algorithms section, API reference, and performance bullets. The API reference entry documents constructor params (including the non-obvious `cluster_selection_epsilon` and the `'manhattan'` metric that an earlier draft omitted), post-fit attributes (`labels_`, `probabilities_`), and the relationship to `find_optimal_clusters` (HDBSCAN is excluded because it determines its own cluster count). The `find_optimal_clusters` example was also corrected to include `'som'` in the algorithm comment, which had been missing.

The tools README now shows correct snake_case API names throughout and presents only current information. `examples/observable/README.md` and `scripts/README.md` were checked and required no changes. GitHub repo description and topics are updated via `gh repo edit` (blocked by permission in automated run — command handed to user for manual execution).

The stale badge, old naming, and historical "Key Findings" are all removed. The post-review pass caught a major omission: the HDBSCAN metric type union in the initial implementation documented only `'euclidean' | 'precomputed'`, silently omitting `'manhattan'` which the implementation fully supports.
