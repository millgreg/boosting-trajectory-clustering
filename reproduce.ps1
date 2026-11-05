$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot
$env:PYTHONPATH = $ProjectRoot
New-Item -ItemType Directory -Force -Path "experiments/results" | Out-Null

function Invoke-Step([string[]]$Arguments) {
    uv run python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Failed: $($Arguments -join ' ')" }
}

Write-Host "[1/6] Data"
Invoke-Step "download_all_data.py"
Invoke-Step "data/tcga_pancan/download_pancan.py"
Invoke-Step "data/gtex/download_gtex.py"

Write-Host "[2/6] Development studies"
Invoke-Step "experiments/run_history_study.py"
Invoke-Step "experiments/run_default_sensitivity.py"
Invoke-Step "experiments/run_proxy_contrast.py"
Invoke-Step "experiments/run_dev_followups.py", "--part", "shap"
Invoke-Step "experiments/run_dev_followups.py", "--part", "homogeneity"

Write-Host "[3/6] Confirmation on held-out datasets"
Invoke-Step "experiments/run_confirmation.py"
Invoke-Step "experiments/analyze_confirmation.py"
Invoke-Step "experiments/run_confirmation_rfc.py"

Write-Host "[4/6] TCGA and GTEx studies"
Invoke-Step "experiments/run_tcga_hierarchy.py"
Invoke-Step "experiments/analyze_tcga_hierarchy.py"
Invoke-Step "experiments/run_tcga_within_cohort.py"
Invoke-Step "experiments/analyze_tcga_within_cohort.py"
Invoke-Step "experiments/run_gtex.py"
Invoke-Step "experiments/analyze_gtex.py"

Write-Host "[5/6] Exploratory studies"
Invoke-Step "experiments/run_target_direction.py"
Invoke-Step "experiments/run_proximity_variants.py"
Invoke-Step "experiments/analyze_proximity_variants.py"
Invoke-Step "experiments/run_unknown_k.py"
Invoke-Step "experiments/analyze_unknown_k.py"

Write-Host "[6/6] Manuscript tables, figures, and numbers"
Invoke-Step "experiments/make_paper_assets.py"

Write-Host "Done. Results are in experiments/results/."
