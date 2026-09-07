# Boosting Trajectories for Supervised Clustering

This repository contains the code to reproduce the experiments and figures for the paper **Boosting Trajectories for Supervised Clustering: Fast Subtype Discovery without Proximity Matrices**.

## Overview

This work discovers latent subtypes within a supervised classification task by clustering the *learning trajectories* of gradient boosting models. Instead of relying on scalar final predictions or spatial features, we extract the sequence of per-tree margin updates across boosting rounds and cluster these temporal trajectories to reveal hidden subpopulation structure in $O(N)$ time.

## Requirements

Dependencies are managed using `uv` and standard `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Quick Start (Usage)

BTC is designed to be fully compatible with the `scikit-learn` API. You can drop it into any existing pipeline where you would normally use `KMeans`.

```python
from sklearn.ensemble import GradientBoostingClassifier
from btc import BTC

# 1. Train a Gradient Booster on a coarse binary proxy task
gb = GradientBoostingClassifier(n_estimators=100, max_depth=1, max_features='sqrt')
gb.fit(X_full, y_coarse_proxy)

# 2. Extract trajectory representations and cluster the target cohort
btc = BTC(booster=gb, n_clusters=4, representation='per_tree_delta')
predicted_subtypes = btc.fit_predict(X_target)
```

## Data Preparation

Before running the full reproduction suite, prepare the biomedical datasets by executing:

```bash
uv run python download_all_data.py
```

This automatically downloads and preprocesses the complex biomedical datasets (TCGA BRCA, TCGA GBMLGG, Paul15, CTG, Mice, Diabetes) from their official sources into `data/`. For complete dataset provenance, ICD-9 diagnostic groupings, and citations, see [data/README.md](data/README.md).

## Reproducing the Paper

To reproduce the entire paper (including running all baselines, ablations, and regenerating the LaTeX tables and PDF figures from scratch), we provide a single entry point script. 

### Running on Windows
```powershell
.\reproduce.ps1
```

### Running on Linux/macOS
```bash
bash reproduce.sh
```

**What this script does:**
1. Runs the canonical BTC and RFC benchmarks (`experiments/run_canonical_benchmark.py`).
2. Computes unsupervised baselines (K-Means, PCA, UMAP) (`experiments/run_unsupervised_baselines.py`).
3. Executes the tree depth ablation (`experiments/run_depth_ablation.py`).
4. Executes the clustering backend ablation & scalability benchmarks (`experiments/run_backend_ablation.py`, `experiments/run_scalability_benchmark.py`).
5. Compiles the LaTeX tables (`experiments/generate_tables.py`, `experiments/generate_ablation_tables.py`).
6. Renders the PDF figures to `paper/figures/` (`experiments/generate_figures.py`).

After running the script, simply compile the LaTeX document in `paper/main.tex` to view the fully populated manuscript.

## Project Structure

```
Miller_BoostingCluster/
├── btc/                        # Core BTC algorithm library
│   ├── __init__.py
│   └── core.py
├── baselines/                  # Benchmark baselines
│   ├── __init__.py
│   └── rfc.py                  # Random Forest Clustering (RFC)
├── data/                       # Biomedical dataset loaders & caches
│   └── README.md
├── experiments/                # Experiment & evaluation suite
│   ├── dataset_loader.py       # Data fetching for all 13 benchmarks
│   ├── run_canonical_benchmark.py
│   ├── run_unsupervised_baselines.py
│   ├── run_depth_ablation.py
│   ├── run_backend_ablation.py
│   ├── run_scalability_benchmark.py
│   ├── run_quick_verification.py
│   ├── generate_tables.py
│   ├── generate_ablation_tables.py
│   └── generate_figures.py
├── paper/                      # TMLR LaTeX source (gitignored)
│   ├── main.tex
│   ├── experiments.tex
│   ├── appendix.tex
│   ├── references.bib
│   ├── tables/                 # Auto-generated LaTeX tables
│   └── figures/                # Auto-generated PDF figures
├── download_all_data.py        # Helper to fetch biomedical data
├── requirements.txt            # Python dependencies
├── setup.py                    # Package build configuration
├── reproduce.sh                # Linux/macOS reproduction entry point
└── reproduce.ps1               # Windows reproduction entry point
```
