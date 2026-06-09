# decision-1 - Cluster-label contract and noise semantics

## Status

Accepted

## Context

The library hosts both partition estimators (KMeans, SpectralClustering,
AgglomerativeClustering, SOM) and density estimators (HDBSCAN). Partition
estimators assign every sample to a cluster; density estimators leave samples
in sparse regions unassigned. For density and partition estimators to
interoperate through the same `LabelVector` type — and for downstream consumers
and validation metrics to interpret labels uniformly — the meaning of a label
value must be fixed library-wide.

## Decision

Cluster labels carry a single authoritative meaning across the library:

- **Non-density estimators** emit a dense labeling in `0..n_clusters-1`. Every
  sample is assigned to exactly one cluster.
- **Density estimators** emit cluster labels `0..n_clusters-1` for assigned
  samples and `-1` for **noise** — samples that belong to no cluster.

`-1` is the only sentinel; it always means "not assigned to any cluster" and is
never a valid cluster index.

### Consequences for internal-validation metrics

Internal-validation metrics measure the cohesion and separation of genuine
clusters, so noise samples must not participate in their distance or dispersion
computations. `silhouette_samples`, `silhouette_score`,
`silhouette_score_subset`, `calinski_harabasz`, `calinski_harabasz_efficient`,
`davies_bouldin`, and `davies_bouldin_efficient` exclude `-1`-labeled samples
before computing any pairwise distance or sum-of-squares.

These metrics remain numerically well-defined at the degenerate boundaries that
noise filtering introduces:

- **All samples are noise** (`every label === -1`): the metrics return a
  defined `0` with no division by zero.
- **One real cluster plus noise**: the metrics return a defined `0` with no
  division by zero.

A genuine degenerate input — fewer than two clusters with **no** noise present
— remains an error, preserving the existing contract for partition estimators.
The distinction is drawn on whether any `-1` label was present in the input.

## Alternatives considered

- **Folding noise into a catch-all cluster.** Rejected: it would corrupt
  internal-validation metrics (noise points are not cohesive) and erase the
  density estimators' core signal that some samples are incidental.
- **A separate noise mask alongside labels.** Rejected: a second parallel array
  is easy to desynchronize and would not interoperate with the existing
  `LabelVector` surface. The `-1` sentinel is the scikit-learn convention and
  needs no new type.
