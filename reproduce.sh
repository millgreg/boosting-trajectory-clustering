#!/usr/bin/env bash
set -e

echo "================================================="
echo " BTC: Reproducing All Paper Results and Figures  "
echo "================================================="

# Get absolute path to project root
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$PROJECT_ROOT"

# Ensure results directories exist
mkdir -p experiments/results

echo ""
echo "[1/6] Running canonical benchmark (BTC vs RFC)..."
uv run python experiments/run_canonical_benchmark.py

echo ""
echo "[2/6] Running unsupervised baselines (K-Means, PCA, UMAP)..."
uv run python experiments/run_unsupervised_baselines.py

echo ""
echo "[3/6] Running depth ablation..."
uv run python experiments/run_depth_ablation.py

echo ""
echo "[4/6] Running clustering backend ablation & scalability..."
uv run python experiments/run_backend_ablation.py
uv run python experiments/run_scalability_benchmark.py

echo ""
echo "[5/6] Generating LaTeX tables..."
uv run python experiments/generate_tables.py
uv run python experiments/generate_ablation_tables.py

echo ""
echo "[6/6] Generating PDF figures..."
uv run python experiments/generate_figures.py

echo ""
echo "================================================="
echo " Done! All artifacts saved to paper/tables and paper/figures. "
echo " You can now compile paper/main.tex "
echo "================================================="
