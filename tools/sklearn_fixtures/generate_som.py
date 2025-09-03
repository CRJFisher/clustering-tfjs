"""Generate reference fixtures for Self-Organizing Maps via MiniSom.

This script generates test fixtures for the SOM implementation using the
MiniSom library as a reference implementation.

Usage
-----
    # From the repo root
    cd tools/sklearn_fixtures
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

    python generate_som.py --out-dir ../../test/fixtures/som

The resulting JSON files feed the Jest tests in `test/clustering/som.reference.test.ts`.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from minisom import MiniSom


DataSpec = Tuple[np.ndarray, str, int]


def generate_datasets(random_state: int = 42) -> List[DataSpec]:
    """Generate test datasets of varying sizes and complexities."""
    # Small datasets for quick testing
    X_iris = datasets.load_iris(return_X_y=True)[0]
    
    # Synthetic datasets
    X_blobs, _ = datasets.make_blobs(
        n_samples=150, centers=3, n_features=4,
        random_state=random_state, cluster_std=0.5
    )
    
    X_moons, _ = datasets.make_moons(
        n_samples=200, noise=0.1, random_state=random_state
    )
    # Add extra dimensions to moons (it's only 2D by default)
    X_moons = np.column_stack([
        X_moons,
        np.random.RandomState(random_state).randn(X_moons.shape[0], 2) * 0.1
    ])
    
    # Medium dataset
    X_digits_subset = datasets.load_digits(return_X_y=True)[0][:500]  # Subset for speed
    
    # Standardize all datasets
    scaler = StandardScaler()
    
    return [
        (scaler.fit_transform(X_iris), "iris", 150),  # 150x4
        (scaler.fit_transform(X_blobs), "blobs", 150),  # 150x4
        (scaler.fit_transform(X_moons), "moons", 200),  # 200x4
        (scaler.fit_transform(X_digits_subset), "digits_subset", 500),  # 500x64
    ]


PARAM_GRID: List[Dict[str, Any]] = [
    # Small grids with different configurations
    {
        "grid_width": 5,
        "grid_height": 5,
        "sigma": 1.0,
        "learning_rate": 0.5,
        "neighborhood_function": "gaussian",
        "topology": "rectangular",
        "num_iteration": 100,
    },
    {
        "grid_width": 10,
        "grid_height": 10,
        "sigma": 1.5,
        "learning_rate": 0.5,
        "neighborhood_function": "gaussian",
        "topology": "rectangular",
        "num_iteration": 200,
    },
    {
        "grid_width": 7,
        "grid_height": 7,
        "sigma": 1.0,
        "learning_rate": 0.3,
        "neighborhood_function": "bubble",
        "topology": "rectangular",
        "num_iteration": 150,
    },
    # Hexagonal topology
    {
        "grid_width": 6,
        "grid_height": 6,
        "sigma": 1.2,
        "learning_rate": 0.5,
        "neighborhood_function": "gaussian",
        "topology": "hexagonal",
        "num_iteration": 100,
    },
]


def calculate_metrics(som: MiniSom, X: np.ndarray) -> Dict[str, float]:
    """Calculate SOM quality metrics."""
    # Quantization error: average distance between samples and their BMUs
    quantization_errors = []
    for x in X:
        bmu = som.winner(x)
        bmu_weight = som.get_weights()[bmu]
        error = np.linalg.norm(x - bmu_weight)
        quantization_errors.append(error)
    
    quantization_error = np.mean(quantization_errors)
    
    # Topographic error: proportion of samples whose BMU and second BMU are not neighbors
    topographic_errors = 0
    for x in X:
        bmu1, bmu2 = som.winner(x), None
        
        # Find second best matching unit
        distances = []
        for i in range(som.get_weights().shape[0]):
            for j in range(som.get_weights().shape[1]):
                if (i, j) != bmu1:
                    dist = np.linalg.norm(x - som.get_weights()[i, j])
                    distances.append((dist, (i, j)))
        
        if distances:
            distances.sort()
            bmu2 = distances[0][1]
            
            # Check if BMU1 and BMU2 are neighbors
            dist = np.sqrt((bmu1[0] - bmu2[0])**2 + (bmu1[1] - bmu2[1])**2)
            if dist > np.sqrt(2):  # Not neighbors (diagonal neighbors allowed)
                topographic_errors += 1
    
    topographic_error = topographic_errors / len(X) if len(X) > 0 else 0
    
    return {
        "quantization_error": float(quantization_error),
        "topographic_error": float(topographic_error),
    }


def dump_fixture(X: np.ndarray, params: Dict[str, Any], out_path: Path) -> None:
    """Train SOM and save fixture."""
    # Initialize SOM
    som = MiniSom(
        x=params["grid_width"],
        y=params["grid_height"],
        input_len=X.shape[1],
        sigma=params["sigma"],
        learning_rate=params["learning_rate"],
        neighborhood_function=params["neighborhood_function"],
        topology=params["topology"],
        random_seed=42,
    )
    
    # Random initialization
    som.random_weights_init(X)
    
    # Train
    som.train_batch(X, params["num_iteration"], verbose=False)
    
    # Get results
    weights = som.get_weights()
    
    # Get BMU for each sample
    bmus = []
    labels = []
    for x in X:
        bmu = som.winner(x)
        bmus.append([int(bmu[0]), int(bmu[1])])  # Convert to native Python int
        # Convert 2D grid position to 1D label
        label = int(bmu[0] * params["grid_width"] + bmu[1])
        labels.append(label)
    
    # Calculate metrics
    metrics = calculate_metrics(som, X)
    
    # Calculate U-matrix (unified distance matrix)
    u_matrix = som.distance_map().T  # Transpose to match our [height, width] convention
    
    fixture = {
        "X": X.astype(float).tolist(),
        "params": {
            "gridWidth": params["grid_width"],
            "gridHeight": params["grid_height"],
            "topology": params["topology"],
            "neighborhood": params["neighborhood_function"],
            "learningRate": params["learning_rate"],
            "radius": params["sigma"],
            "numEpochs": params["num_iteration"],
            "randomState": 42,
        },
        "weights": weights.astype(float).tolist(),  # [height, width, features]
        "labels": labels,  # 1D cluster assignments
        "bmus": bmus,  # 2D grid positions
        "uMatrix": u_matrix.astype(float).tolist(),
        "metrics": metrics,
    }
    
    out_path.write_text(json.dumps(fixture, indent=2))
    print(f"  Wrote {out_path.name} - QE: {metrics['quantization_error']:.4f}, TE: {metrics['topographic_error']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = generate_datasets()
    
    print("Generating SOM fixtures...")
    print("-" * 50)
    
    for X, ds_name, n_samples in datasets:
        print(f"\nDataset: {ds_name} ({n_samples}x{X.shape[1]})")
        
        # Only use parameter configs that make sense for the dataset size
        suitable_params = []
        for p in PARAM_GRID:
            grid_size = p["grid_width"] * p["grid_height"]
            # Grid should be smaller than the number of samples
            if grid_size <= n_samples:
                suitable_params.append(p)
        
        for p in suitable_params:
            fname = (
                f"{ds_name}_{p['grid_width']}x{p['grid_height']}_"
                f"{p['neighborhood_function']}_{p['topology']}.json"
            )
            dump_fixture(X, p, args.out_dir / fname)
    
    print("\n" + "=" * 50)
    print(
        f"Fixtures written to {args.out_dir}\n"
        f"Total files: {len(list(args.out_dir.glob('*.json')))}"
    )


if __name__ == "__main__":
    main()