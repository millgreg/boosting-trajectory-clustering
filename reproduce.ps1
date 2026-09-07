$ErrorActionPreference = "Stop"

Write-Host "================================================="
Write-Host " BTC: Reproducing All Paper Results and Figures  "
Write-Host "================================================="

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = $ProjectRoot

New-Item -ItemType Directory -Force -Path "experiments/results" | Out-Null

Write-Host "`n[1/6] Running canonical benchmark (BTC vs RFC)..."
uv run python experiments/run_canonical_benchmark.py

Write-Host "`n[2/6] Running unsupervised baselines (K-Means, PCA, UMAP)..."
uv run python experiments/run_unsupervised_baselines.py

Write-Host "`n[3/6] Running depth ablation..."
uv run python experiments/run_depth_ablation.py

Write-Host "`n[4/6] Running clustering backend ablation & scalability..."
uv run python experiments/run_backend_ablation.py
uv run python experiments/run_scalability_benchmark.py

Write-Host "`n[5/6] Generating LaTeX tables..."
uv run python experiments/generate_tables.py
uv run python experiments/generate_ablation_tables.py
uv run python experiments/run_diagnostic_audit.py

Write-Host "`n[6/6] Generating PDF figures..."
uv run python experiments/generate_figures.py

Write-Host "`n================================================="
Write-Host " Done! All artifacts saved to paper/tables and paper/figures. "
Write-Host " You can now compile paper/main.tex "
Write-Host "================================================="
