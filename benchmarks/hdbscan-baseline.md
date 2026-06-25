# HDBSCAN float64-JS Baseline (task-54)

Pre-migration baseline of the all-JS HDBSCAN pipeline, captured on the `cpu`
backend (median of 3 runs per config). HDBSCAN is backend-independent at
this point; task-54.3+ move the front-half onto tfjs and the post-migration run
is diffed against this file.

## Results

| Config | n × d | Median (ms) | Min (ms) | Max (ms) | Memory (MB) |
|--------|-------|-------------|----------|----------|-------------|
| small | 100×10 | 4.87 | 1.83 | 7.66 | 0.00 |
| medium | 1000×50 | 226.95 | 225.96 | 244.94 | 0.00 |
| hdbscan_n2000_d2 | 2000×2 | 937.51 | 891.60 | 1038.09 | 0.00 |
| hdbscan_n2000_d16 | 2000×16 | 932.30 | 901.83 | 942.26 | 0.00 |
| hdbscan_n2000_d64 | 2000×64 | 1082.11 | 1067.72 | 1117.10 | 0.00 |
| hdbscan_n2000_d128 | 2000×128 | 1182.55 | 1143.35 | 1276.97 | 0.00 |
| hdbscan_n5000_d16 | 5000×16 | 6284.35 | 6276.77 | 6894.00 | 0.00 |
| hdbscan_n5000_d128 | 5000×128 | 8887.47 | 8297.86 | 8943.51 | 0.00 |
