#!/usr/bin/env bash
set -e
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"
mkdir -p experiments/results

echo "[1/6] Data"
uv run python download_all_data.py
uv run python data/tcga_pancan/download_pancan.py
uv run python data/gtex/download_gtex.py

echo "[2/6] Development studies"
uv run python experiments/run_history_study.py
uv run python experiments/run_default_sensitivity.py
uv run python experiments/run_proxy_contrast.py
uv run python experiments/run_dev_followups.py --part shap
uv run python experiments/run_dev_followups.py --part homogeneity

echo "[3/6] Confirmation on held-out datasets"
uv run python experiments/run_confirmation.py
uv run python experiments/analyze_confirmation.py
uv run python experiments/run_confirmation_rfc.py

echo "[4/6] TCGA and GTEx studies"
uv run python experiments/run_tcga_hierarchy.py
uv run python experiments/analyze_tcga_hierarchy.py
uv run python experiments/run_tcga_within_cohort.py
uv run python experiments/analyze_tcga_within_cohort.py
uv run python experiments/run_gtex.py
uv run python experiments/analyze_gtex.py

echo "[5/6] Exploratory studies"
uv run python experiments/run_target_direction.py
uv run python experiments/run_proximity_variants.py
uv run python experiments/analyze_proximity_variants.py
uv run python experiments/run_unknown_k.py
uv run python experiments/analyze_unknown_k.py
uv run python experiments/run_alignment.py
uv run python experiments/analyze_alignment.py
uv run python experiments/proxy_alignment.py
uv run python experiments/run_capacity.py
uv run python experiments/analyze_capacity.py

echo "[6/6] Manuscript tables, figures, and numbers"
uv run python experiments/make_paper_assets.py

echo "Done. Results are in experiments/results/."
