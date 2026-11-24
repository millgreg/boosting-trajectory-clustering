# Boosting Trajectory Clustering: A Pre-Specified Comparison with Supervised and Unsupervised Clustering

Code for a study of proxy-guided subtype discovery: using a coarse binary label (the proxy) to find finer subtypes within a target cohort. It includes Boosting Trajectory Clustering (BTC), which clusters samples by the per-tree margin updates of a gradient boosting classifier, and compares it with Random Forest Clustering (RFC), SHAP-based clustering, random-forest leaf embeddings, and unsupervised baselines.

## Main findings

- On 40 held-out OpenML datasets with random class groupings, the pre-specified primary claim, that BTC outperforms TreeSHAP clustering, was not confirmed: BTC performed on par with standardized TreeSHAP clustering and UMAP (mean ARI 0.200, 0.205 and 0.204). Spectral RFC scored highest (0.252, exploratory).
- On TCGA PanCanAtlas data, plain K-Means outperformed every supervised tree representation, both for cancer types within organ systems and for molecular subtypes within cancer types. Proxies measured within a cancer type (stage, grade, survival and others) did not help either, except hormone-receptor status in breast cancer.
- On GTEx, where the finer labels are anatomical sampling sites rather than results of clustering, plain K-Means again outperformed every supervised tree representation (proxy: one tissue versus all others).
- Pseudo-residual histories did not cluster clearly better than the final margin, learning-speed summaries clustered worse, and tree order mattered little.
- In a controlled study with proxies built from the subtypes, supervision helped only when the proxy was strongly aligned with the subtypes; real clinical proxies were far less aligned.
- Tree-based representations are unaffected by feature scaling, whereas K-Means results change substantially between raw and standardized features.

Confirmatory analyses followed a protocol frozen before the run; see `experiments/confirmation/PROTOCOL.md`.

## Installation

```bash
pip install -r requirements-lock.txt
```

`requirements-lock.txt` records the full environment used for the reported results (Python 3.14), including indirect dependencies. `requirements.txt` lists only the direct dependencies and is enough to use BTC. Install `pytest` to run the tests with `python -m pytest tests`.

## Usage

```python
from sklearn.ensemble import GradientBoostingClassifier
from btc import BTC

booster = GradientBoostingClassifier(n_estimators=100, max_depth=1, max_features='sqrt')
booster.fit(X_full, y_proxy)
labels = BTC(booster=booster, n_clusters=4).fit_predict(X_target)
```

`baselines.FactorizedRFC` computes spectral Random Forest Clustering exactly without forming the N x N proximity matrix.

## Reproducing the results

`reproduce.sh` (Linux/macOS) and `reproduce.ps1` (Windows) run every step in order:

1. **Data:** `download_all_data.py` prepares the local biomedical benchmarks, and `data/tcga_pancan/download_pancan.py` and `data/gtex/download_gtex.py` fetch the TCGA PanCanAtlas and GTEx files. OpenML datasets download automatically.
2. **Development studies** on the 13 benchmarks and OpenML-CC18: `run_history_study.py`, `run_default_sensitivity.py`, `run_proxy_contrast.py`, and `run_dev_followups.py`.
3. **Confirmation** on 40 held-out datasets: `run_confirmation.py`, analyzed by `analyze_confirmation.py`, plus the exploratory `run_confirmation_rfc.py`.
4. **TCGA and GTEx studies:** `run_tcga_hierarchy.py`, `run_tcga_within_cohort.py`, and `run_gtex.py`, each with its `analyze_` script.
5. **Exploratory studies:** `run_target_direction.py`, `run_proximity_variants.py` (Breiman, out-of-bag, and RF-GAP proximities), `run_unknown_k.py` (number of subtypes estimated), `run_alignment.py` (proxies built from the subtypes with controlled label noise), and `run_capacity.py` (larger boosted models), with their `analyze_` scripts, plus `proxy_alignment.py` (alignment of the real TCGA proxies).
6. **Manuscript assets:** `make_paper_assets.py` generates every table, figure, and reported number.

The development studies were exploratory. For the confirmation, TCGA, GTEx, alignment, proximity, unknown-K, and model-size studies, the analysis plan is written in the docstring of the run script and was committed before the corresponding results existed; `run_target_direction.py` and `proxy_alignment.py` are exploratory and had no plan. Results are written to `experiments/results/`.

## Project structure

```
btc/                    BTC and trajectory extraction (scikit-learn, XGBoost, LightGBM)
baselines/              Spectral Random Forest Clustering (dense and factorized) and proximity variants
data/                   Download and preprocessing scripts for the local datasets
experiments/            Study scripts, analyses, and the frozen confirmation protocol
tests/                  Regression tests
```

## License

MIT; see `LICENSE`.
