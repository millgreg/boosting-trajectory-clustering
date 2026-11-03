$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot
$env:PYTHONPATH = $ProjectRoot
New-Item -ItemType Directory -Force -Path "experiments/results" | Out-Null

function Invoke-Step([string[]]$Arguments) {
    uv run python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Failed: $($Arguments -join ' ')" }
}

Write-Host "[1/4] Data"
Invoke-Step "download_all_data.py"
Invoke-Step "data/tcga_pancan/download_pancan.py"

Write-Host "[2/4] Development studies"
Invoke-Step "experiments/run_history_study.py"
Invoke-Step "experiments/run_default_sensitivity.py"
Invoke-Step "experiments/run_proxy_contrast.py"
Invoke-Step "experiments/run_dev_followups.py", "--part", "shap"
Invoke-Step "experiments/run_dev_followups.py", "--part", "homogeneity"

Write-Host "[3/4] Confirmation on held-out datasets"
Invoke-Step "experiments/run_confirmation.py"
Invoke-Step "experiments/analyze_confirmation.py"
Invoke-Step "experiments/run_confirmation_rfc.py"

Write-Host "[4/4] TCGA studies"
Invoke-Step "experiments/run_tcga_hierarchy.py"
Invoke-Step "experiments/analyze_tcga_hierarchy.py"
Invoke-Step "experiments/run_tcga_within_cohort.py"
Invoke-Step "experiments/analyze_tcga_within_cohort.py"

Write-Host "Done. Results are in experiments/results/."
