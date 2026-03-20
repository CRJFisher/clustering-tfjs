---
id: TASK-42
title: Fix findOptimalClusters combined scoring and add proper methods
status: In Progress
assignee: []
created_date: '2026-03-20'
updated_date: '2026-03-20 15:44'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The findOptimalClusters default combined score adds raw unnormalized metrics (silhouette [-1,1] + calinskiHarabasz [0,thousands] - daviesBouldin [0,infinity]), making Calinski-Harabasz completely dominate. The function also lacks elbow method, gap statistic, and proper silhouette-only method. Validation metrics have additional issues: silhouetteScore NaN when both a and b are 0, daviesBouldin returns Infinity for coincident centroids, no validation that labels length matches data length, no noise label (-1) handling, no per-sample silhouette scores exposed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Combined score uses normalized metrics so all three contribute meaningfully
- [x] #2 Elbow method implemented using WSS/inertia curve knee detection
- [x] #3 Silhouette-only method implemented
- [x] #4 silhouetteScore returns 0 (not NaN) when a==0 and b==0
- [x] #5 daviesBouldin handles coincident centroids matching sklearn behavior
- [x] #6 Labels length validated against data rows in all validation functions
- [x] #7 Per-sample silhouette scores exposed as silhouetteSamples function
- [x] #8 Missing metrics: ARI and NMI added for supervised evaluation
<!-- AC:END -->

## Implementation Plan

1. Fix silhouetteScore NaN (AC#4): add `a === 0 && b === 0` guard in both silhouetteScore and silhouetteScoreSubset
2. Fix daviesBouldin coincident centroids (AC#5): split distance===0 handling into zero-dispersion (skip/continue) vs nonzero-dispersion (Infinity)
3. Add labels-length validation (AC#6): create shared validateLabelsLength helper, add to all 6 validation functions
4. Extract silhouetteSamples (AC#7): refactor silhouetteScore to delegate to new silhouetteSamples function
5. Implement ARI and NMI (AC#8): create contingency table builder, adjusted_rand_index.ts, normalized_mutual_info.ts
6. Normalize combined scoring (AC#1): two-pass normalization with min-max to [0,1] range
7. Add elbow and silhouette methods (AC#2+3): create computeWss.ts, kneedle.ts, add method option to findOptimalClusters

## Implementation Notes

### Approach
- Bottom-up: fixed validation bugs first (AC#4-6), then added new capabilities (AC#7-8), then overhauled scoring (AC#1-3)
- All changes are additive and backward compatible; no existing public API signatures changed

### Features implemented
- **Normalized scoring** (AC#1): `normalizeAndScoreEvaluations()` helper normalizes silhouette via fixed-range mapping `(s+1)/2`, CH and DB via min-max across evaluations. Combined score is the mean of normalized metrics in [0,1].
- **Elbow method** (AC#2): New `computeWss()` utility (with KMeans inertia_ optimization). Kneedle algorithm (`findKnee()`) detects knee in WSS-vs-k curve by finding max deviation from the diagonal.
- **Silhouette-only method** (AC#3): When `method: 'silhouette'`, only silhouette is computed; combinedScore equals raw silhouette.
- **Method option**: `FindOptimalClustersOptions.method` accepts `'combined' | 'elbow' | 'silhouette'`. Custom `scoringFunction` still overrides method.
- **silhouetteSamples** (AC#7): Returns per-sample `number[]`; `silhouetteScore` is now a thin wrapper.
- **ARI** (AC#8): Contingency-table-based adjusted rand index matching sklearn conventions (denominator 0 → return 0).
- **NMI** (AC#8): Supports `'arithmetic' | 'geometric' | 'min' | 'max'` averaging methods. Handles zero-entropy edge cases per sklearn.
- **Labels validation** (AC#6): Shared `validateLabelsLength()` in `src/validation/validate.ts`, called at top of all 6 validation functions.

### Technical decisions
- Kneedle uses deviation from the diagonal line connecting first/last normalized points; concave curves have negative deviations, so we negate for comparison
- daviesBouldin coincident centroids: when both dispersions and centroid distance are 0, we `continue` (skip pair) matching sklearn's nanmax behavior
- ARI: singleton-per-point clusterings return 0 (sklearn convention, denominator is 0)
- NMI: when both entropies are 0, return 1.0 (trivially identical partitions)

### New files
- `src/validation/validate.ts` - shared labels-length validation
- `src/validation/contingency.ts` - contingency table builder for ARI/NMI
- `src/validation/adjusted_rand_index.ts` - ARI implementation
- `src/validation/normalized_mutual_info.ts` - NMI implementation
- `src/utils/computeWss.ts` - within-cluster sum of squares
- `src/utils/kneedle.ts` - knee/elbow detection algorithm

### Modified files
- `src/validation/silhouette.ts` - refactored to silhouetteSamples + NaN fix
- `src/validation/davies_bouldin.ts` - coincident centroid fix + validation
- `src/validation/calinski_harabasz.ts` - labels validation
- `src/utils/findOptimalClusters.ts` - normalized scoring + method option + WSS + elbow
- `src/validation/index.ts`, `src/utils/index.ts`, `src/index.ts` - new exports
