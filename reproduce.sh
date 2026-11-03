#!/usr/bin/env bash
set -e
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"
mkdir -p experiments/results

echo "[1/4] Data"
uv run python download_all_data.py
uv run python data/tcga_pancan/download_pancan.py

echo "[2/4] Development studies"
uv run python experiments/run_history_study.py
uv run python experiments/run_default_sensitivity.py
uv run python experiments/run_proxy_contrast.py
uv run python experiments/run_dev_followups.py --part shap
uv run python experiments/run_dev_followups.py --part homogeneity

echo "[3/4] Confirmation on held-out datasets"
uv run python experiments/run_confirmation.py
uv run python experiments/analyze_confirmation.py
uv run python experiments/run_confirmation_rfc.py

echo "[4/4] TCGA studies"
uv run python experiments/run_tcga_hierarchy.py
uv run python experiments/analyze_tcga_hierarchy.py
uv run python experiments/run_tcga_within_cohort.py
uv run python experiments/analyze_tcga_within_cohort.py

echo "Done. Results are in experiments/results/."
