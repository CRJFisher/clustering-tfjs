# SOM benchmarking against MiniSom

This guide describes how the Self-Organizing Map implementation is validated for
numerical correctness against [MiniSom](https://github.com/JustGlowing/minisom),
and how to regenerate the reference fixtures.

## Two training paths

The library has two distinct SOM training paths, validated separately.

- **Production path — `SOM.fit` (online mini-batch).** The default training used
  by applications. It shuffles the data each epoch, processes mini-batches, and
  applies a normalized batch update `w += lr · Σ(h·(x − w)) / Σ(h)` with
  per-epoch exponential decay. It is tuned for GPU/tensor execution and streaming
  (`partial_fit`). Its invariants — shapes, determinism under a fixed seed,
  convergence, clustering, persistence — are covered by `src/clustering/som.test.ts`
  and `src/clustering/som_hexagonal.test.ts`.

- **Reference path — `train_minisom_reference` (`src/clustering/som_reference_training.ts`).**
  A tensor-free transcription of MiniSom's `train_batch` used only for numeric
  validation. It reproduces MiniSom's output to floating-point precision, which
  lets the reference suite (`src/clustering/som_reference.test.ts`) assert tight
  tolerances. It is not part of the production API.

These are different algorithms. The reference path exists so the SOM can be held
to rigorous parity bounds without constraining how the production path trains.

## What the reference path replicates

MiniSom's `train_batch(data, num_iteration)` is deterministic online-sequential
training, not a batch-map SOM:

- **Sample order.** Iteration `t = 0 … num_iteration − 1` uses sample
  `data[t mod n_samples]`. There is no shuffling; the only stochastic input to
  the whole procedure is the initial weight grid.
- **Update.** Each iteration finds the best-matching unit (BMU) and updates every
  neuron: `w += η(t) · g(t) · (x − w)`, where `g` is the neighborhood centered on
  the BMU. There is no influence-sum normalization.
- **Decay (asymptotic, per iteration).** Both learning rate and neighborhood
  radius follow `p(t) = p₀ / (1 + t / (num_iteration / 2))`.

Because the procedure is deterministic once the initial weights are fixed,
injecting identical initial weights into MiniSom (at fixture-generation time) and
into the reference trainer (at test time) removes initialization and RNG as
variables, so the two match to ~1e-12.

### Neighborhood functions

The reference trainer matches MiniSom's exact definitions, centered on BMU
`(c_x, c_y)` with radius `σ`:

- **Gaussian** — separable product of two 1-D gaussians over the grid
  coordinates, equal to `exp(−((Δx)² + (Δy)²) / (2σ²))`.
- **Bubble** — separable open-interval box on the integer grid indices:
  `1` where `c − σ < index < c + σ` (strict on both sides), else `0`. This is a
  box, not a Euclidean disc.
- **Mexican hat (Ricker)** — `exp(−p / d) · (1 − 2p / d)` with `p = (Δx)² + (Δy)²`
  and `d = 2σ²`.

### Grid coordinates and topology

Neuron coordinates `(x, y)` drive the gaussian and mexican-hat neighborhoods and
the hexagonal topographic error.

- **Rectangular** — integer indices: `x = col`, `y = row`.
- **Hexagonal** — rows are shifted and scaled to a unit-spaced hex layout:
  `y = row · 0.8660254`, and `x = col − 0.5` for rows where
  `(grid_height − 1 − row)` is even (matching MiniSom's `_xx[::-2] -= 0.5`),
  otherwise `x = col`.

MiniSom stores weights in `[x = width][y = height][features]` order and `winner()`
returns `(x, y) = (col, row)`. The reference trainer works internally in that
native orientation and transposes at its boundary so all public inputs and
outputs use the library's `[grid_height][grid_width][n_features]` convention.

### Connectivity

Rectangular topology uses **8-connectivity** (a neuron's neighbors include
diagonals). The U-matrix sums distances to up to 8 neighbors; topographic error
treats two neurons as adjacent when their grid-index distance is `≤ √2` (the
error threshold is `> 1.42`). Hexagonal topology uses **6-connectivity**, and
topographic error treats neurons as adjacent when their Euclidean hex-coordinate
distance is `1`. Both the production code and the reference trainer use these same
conventions, which is why the metrics agree.

## Fixtures

Reference fixtures live in `__fixtures__/som/*.json`, one per
dataset × grid × neighborhood × topology combination. Each file contains:

| Field | Meaning |
| --- | --- |
| `name` | Fixture identifier (the file stem). |
| `X` | Standardized input data `[n_samples][n_features]`. |
| `params.grid_width`, `params.grid_height` | Grid dimensions. |
| `params.topology` | `rectangular` or `hexagonal`. |
| `params.neighborhood` | `gaussian`, `bubble`, or `mexican_hat`. |
| `params.learning_rate` | Initial learning rate `η₀`. |
| `params.radius` | Initial neighborhood radius `σ₀` (MiniSom `sigma`). |
| `params.num_iteration` | Number of per-sample updates (MiniSom `num_iteration`, **not** epochs). |
| `params.random_state` | Seed used to derive the injected initial weights. |
| `initial_weights` | Initial weight grid `[grid_height][grid_width][n_features]`, injected into both implementations. |
| `weights` | Final weight grid after training, same shape. |
| `bmus` | Per-sample BMU grid coordinates as `[row, col]`. |
| `labels` | Per-sample flat labels (`row · grid_width + col`). |
| `u_matrix` | Normalized inter-neuron distance map `[grid_height][grid_width]`. |
| `metrics.quantization_error` | MiniSom `quantization_error(X)`. |
| `metrics.topographic_error` | MiniSom `topographic_error(X)`. |

Coverage spans the `iris`, `blobs`, `moons`, and `digits_subset` datasets; square
and non-square grids (the complementary 8×4 and 4×8 grids expose any axis
transpose error); both topologies; and all three neighborhood functions.

## Tolerances

| Quantity | Bound | Rationale |
| --- | --- | --- |
| Weights (per element) | `1e-9` | Continuous in the matched weights; observed parity ~1e-12. |
| Quantization error | `1e-9` | Continuous function of the matched weights. |
| U-matrix (per element) | `1e-9` | Continuous function of the matched weights. |
| Topographic error | `≤ 1%` relative (exact when reference is 0) | Discrete metric: the second-nearest-neuron choice is discontinuous at distance degeneracies, where a sub-1e-9 weight difference flips one sample's classification. |
| BMU indices, labels | exact | Integer argmins over matched weights. |

The weight, QE, and U-matrix bounds are far tighter than the original loose
tolerances (per-element weight average difference `< 2.0`, QE relative error
`< 60%`). No reference cases are skipped.

## Regenerating fixtures

Fixture regeneration is a manual developer step. It is **not** run in CI: it
requires a Python toolchain with MiniSom and must stay deterministic against a
pinned dependency set. CI consumes the committed JSON fixtures and runs the Jest
reference suite as the gate.

```sh
cd tools/sklearn_fixtures
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # numpy, scikit-learn, minisom (pinned)
python generate_som.py --out-dir ../../__fixtures__/som
```

`generate_som.py` seeds a deterministic initial weight grid, injects it into
MiniSom, trains with `train_batch`, and writes each fixture in the schema above.
Regenerate only when the matched algorithm or the scenario set changes, then run
the reference suite to confirm parity holds:

```sh
npx jest src/clustering/som_reference.test.ts
```
