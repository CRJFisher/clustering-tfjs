"""Generate numeric reference fixtures for Self-Organizing Maps via MiniSom.

These fixtures drive the rigorous SOM reference suite
(`src/clustering/som_reference.test.ts`), which asserts that the TypeScript
reference trainer (`src/clustering/som_reference_training.ts`) reproduces
MiniSom's deterministic ``train_batch`` to floating-point precision.

Reproducibility hinges on two things:

1. ``train_batch`` is deterministic given fixed initial weights — it iterates
   samples sequentially (``data[t % n]``), applies a single-sample update over
   the whole grid, and decays learning rate and sigma asymptotically as
   ``param / (1 + t / (num_iteration / 2))``. No RNG is used after weight init.
2. The initial weight grid is injected, not randomly generated, so both MiniSom
   and the TypeScript trainer start from byte-identical weights. Each fixture
   stores ``initial_weights`` for the TypeScript side to inject verbatim.

Axis convention: MiniSom stores weights as ``[x=width][y=height][features]`` and
``winner()`` returns ``(x, y) = (col, row)``. Every grid-shaped field is
transposed to the library's ``[height][width][features]`` convention before it
is written, and BMU/label indices are converted to ``[row, col]`` /
``row * grid_width + col``.

Usage (developer-only; not run in CI)
-------------------------------------
    # From the repo root
    cd tools/sklearn_fixtures
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

    python generate_som.py --out-dir ../../__fixtures__/som
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from minisom import MiniSom
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

DataSpec = Tuple[np.ndarray, str]

# Seed used to derive the injected initial weights for every fixture.
INIT_SEED = 42


def generate_datasets(random_state: int = 42) -> List[DataSpec]:
    """Standardized test datasets of varying sizes and dimensionalities."""
    x_iris = datasets.load_iris(return_X_y=True)[0]

    x_blobs, _ = datasets.make_blobs(
        n_samples=150, centers=3, n_features=4, random_state=random_state, cluster_std=0.5
    )

    x_moons, _ = datasets.make_moons(n_samples=200, noise=0.1, random_state=random_state)
    # make_moons is 2D; pad to 4D with small seeded noise so the SOM sees more structure.
    x_moons = np.column_stack(
        [x_moons, np.random.RandomState(random_state).randn(x_moons.shape[0], 2) * 0.1]
    )

    x_digits = datasets.load_digits(return_X_y=True)[0][:500]  # subset for speed

    scaler = StandardScaler()
    return [
        (scaler.fit_transform(x_iris), "iris"),
        (scaler.fit_transform(x_blobs), "blobs"),
        (scaler.fit_transform(x_moons), "moons"),
        (scaler.fit_transform(x_digits), "digits_subset"),
    ]


# Each config covers a combination of grid shape, neighborhood, and topology.
# Coverage spans gaussian/bubble/mexican_hat x rectangular/hexagonal, square and
# non-square grids (8x4 and 4x8 deliberately expose axis-transpose bugs).
PARAM_GRID: List[Dict[str, Any]] = [
    {"grid_width": 5, "grid_height": 5, "sigma": 1.0, "learning_rate": 0.5,
     "neighborhood_function": "gaussian", "topology": "rectangular", "num_iteration": 100},
    {"grid_width": 10, "grid_height": 10, "sigma": 1.5, "learning_rate": 0.5,
     "neighborhood_function": "gaussian", "topology": "rectangular", "num_iteration": 200},
    {"grid_width": 7, "grid_height": 7, "sigma": 1.0, "learning_rate": 0.3,
     "neighborhood_function": "bubble", "topology": "rectangular", "num_iteration": 150},
    {"grid_width": 6, "grid_height": 6, "sigma": 1.2, "learning_rate": 0.5,
     "neighborhood_function": "gaussian", "topology": "hexagonal", "num_iteration": 100},
    {"grid_width": 6, "grid_height": 6, "sigma": 1.0, "learning_rate": 0.5,
     "neighborhood_function": "mexican_hat", "topology": "rectangular", "num_iteration": 120},
    {"grid_width": 6, "grid_height": 6, "sigma": 1.0, "learning_rate": 0.5,
     "neighborhood_function": "mexican_hat", "topology": "hexagonal", "num_iteration": 120},
    {"grid_width": 8, "grid_height": 4, "sigma": 1.0, "learning_rate": 0.5,
     "neighborhood_function": "gaussian", "topology": "rectangular", "num_iteration": 150},
    {"grid_width": 4, "grid_height": 8, "sigma": 1.0, "learning_rate": 0.5,
     "neighborhood_function": "bubble", "topology": "rectangular", "num_iteration": 150},
]


def make_initial_weights(
    X: np.ndarray, grid_width: int, grid_height: int, seed: int
) -> np.ndarray:
    """Deterministic initial weight grid in MiniSom native (width, height, features) order.

    Picks random data rows (the same family of initialization MiniSom's
    ``random_weights_init`` uses) via an explicit seeded generator so the grid is
    fully reproducible and can be injected into both implementations.
    """
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, X.shape[0], size=grid_width * grid_height)
    return X[idx].reshape(grid_width, grid_height, X.shape[1]).astype(float).copy()


def dump_fixture(X: np.ndarray, params: Dict[str, Any], name: str, out_path: Path) -> None:
    """Train MiniSom from injected initial weights and write the fixture JSON."""
    grid_width = params["grid_width"]
    grid_height = params["grid_height"]
    n_features = X.shape[1]

    som = MiniSom(
        x=grid_width,
        y=grid_height,
        input_len=n_features,
        sigma=params["sigma"],
        learning_rate=params["learning_rate"],
        neighborhood_function=params["neighborhood_function"],
        topology=params["topology"],
        random_seed=INIT_SEED,
    )

    # Inject deterministic initial weights (native [width][height][features]).
    initial_weights_native = make_initial_weights(X, grid_width, grid_height, INIT_SEED)
    som._weights = initial_weights_native.copy()

    som.train_batch(X, params["num_iteration"], verbose=False)

    # Transpose grid-shaped outputs to library [height][width][...] convention.
    initial_weights = initial_weights_native.transpose(1, 0, 2)
    weights = som.get_weights().transpose(1, 0, 2)
    u_matrix = som.distance_map().T

    bmus: List[List[int]] = []
    labels: List[int] = []
    for x in X:
        col, row = som.winner(x)  # MiniSom returns (x=col, y=row)
        bmus.append([int(row), int(col)])
        labels.append(int(row) * grid_width + int(col))

    fixture = {
        "name": name,
        "X": X.astype(float).tolist(),
        "params": {
            "grid_width": grid_width,
            "grid_height": grid_height,
            "topology": params["topology"],
            "neighborhood": params["neighborhood_function"],
            "learning_rate": params["learning_rate"],
            "radius": params["sigma"],
            "num_iteration": params["num_iteration"],
            "random_state": INIT_SEED,
        },
        "initial_weights": initial_weights.astype(float).tolist(),
        "weights": weights.astype(float).tolist(),
        "bmus": bmus,
        "labels": labels,
        "u_matrix": u_matrix.astype(float).tolist(),
        "metrics": {
            "quantization_error": float(som.quantization_error(X)),
            "topographic_error": float(som.topographic_error(X)),
        },
    }

    out_path.write_text(json.dumps(fixture, indent=2))
    print(
        f"  Wrote {out_path.name} - QE: {fixture['metrics']['quantization_error']:.4f}, "
        f"TE: {fixture['metrics']['topographic_error']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating SOM fixtures...")
    print("-" * 50)

    for X, ds_name in generate_datasets():
        print(f"\nDataset: {ds_name} ({X.shape[0]}x{X.shape[1]})")
        for p in PARAM_GRID:
            # Skip grids larger than the dataset.
            if p["grid_width"] * p["grid_height"] > X.shape[0]:
                continue
            name = (
                f"{ds_name}_{p['grid_width']}x{p['grid_height']}_"
                f"{p['neighborhood_function']}_{p['topology']}"
            )
            dump_fixture(X, p, name, args.out_dir / f"{name}.json")

    print("\n" + "=" * 50)
    print(
        f"Fixtures written to {args.out_dir}\n"
        f"Total files: {len(list(args.out_dir.glob('*.json')))}"
    )


if __name__ == "__main__":
    main()
