# HDBSCAN float64-JS Baseline (task-54)

Pre-migration baseline of the all-JS HDBSCAN pipeline, captured on the `cpu`
backend (median of 3 runs per config). HDBSCAN is backend-independent at
this point; task-54.3+ move the front-half onto tfjs and the post-migration run
is diffed against this file.

## Results

| Config | n × d | Median (ms) | Min (ms) | Max (ms) | Memory (MB) |
|--------|-------|-------------|----------|----------|-------------|
| small | 100×10 | 3.18 | 1.82 | 7.55 | 0.00 |
| medium | 1000×50 | 234.43 | 227.62 | 243.64 | 0.00 |
| hdbscan_n2000_d2 | 2000×2 | 924.73 | 895.73 | 951.85 | 0.00 |
| hdbscan_n2000_d16 | 2000×16 | 932.62 | 899.56 | 933.24 | 0.00 |
| hdbscan_n2000_d64 | 2000×64 | 1060.18 | 1051.91 | 1070.81 | 0.00 |
| hdbscan_n2000_d128 | 2000×128 | 1238.21 | 1224.36 | 1249.14 | 0.00 |
| hdbscan_n5000_d16 | 5000×16 | 6715.10 | 6196.30 | 6744.97 | 0.00 |
| hdbscan_n5000_d128 | 5000×128 | 8364.99 | 8274.51 | 8551.76 | 0.00 |
