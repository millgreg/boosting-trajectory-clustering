# Boosting Trajectories for Supervised Clustering

Code for a study of proxy-guided subtype discovery: using a coarse binary label (the proxy) to find finer subtypes within a target cohort. It includes Boosting Trajectory Clustering (BTC), which clusters samples by the per-tree margin updates of a gradient boosting classifier, and compares it with Random Forest Clustering (RFC), SHAP-based clustering, random-forest leaf embeddings, and unsupervised baselines.

## Main findings

- On 40 held-out OpenML datasets with random class groupings, BTC performed on par with standardized TreeSHAP clustering and UMAP (mean ARI 0.200, 0.205 and 0.204). Spectral RFC scored highest (0.252, exploratory).
- On TCGA PanCanAtlas data, plain K-Means outperformed every supervised tree representation, both for cancer types within organ systems and for molecular subtypes within cancer types. Proxies measured within a cancer type (stage, grade, survival and others) did not help either, except hormone-receptor status in breast cancer.
- Boosting histories carry little beyond the final margin and the tree splits: pseudo-residuals, learning speed, and tree order add little or nothing.
- Tree-based representations are unaffected by feature scaling, whereas K-Means results change substantially between raw and standardized features.

Confirmatory analyses followed a protocol frozen before the run; see `experiments/confirmation/PROTOCOL.md`.

## Installation

```bash
pip install -r requirements.txt
```

`requirements.txt` pins the versions used for the reported results, and `requirements-lock.txt` records the full environment (Python 3.14). Install `pytest` to run the tests with `python -m pytest tests`.

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

1. **Data:** `download_all_data.py` prepares the local biomedical benchmarks, and `data/tcga_pancan/download_pancan.py` fetches the TCGA PanCanAtlas files. OpenML datasets download automatically.
2. **Development studies** on the 13 benchmarks and OpenML-CC18: `run_history_study.py`, `run_default_sensitivity.py`, `run_proxy_contrast.py`, and `run_dev_followups.py`.
3. **Confirmation** on 40 held-out datasets: `run_confirmation.py`, analysed by `analyze_confirmation.py`, plus the exploratory `run_confirmation_rfc.py`.
4. **TCGA studies:** `run_tcga_hierarchy.py` and `run_tcga_within_cohort.py`, each with its `analyze_` script.
5. **Exploratory studies:** `run_target_direction.py`, `run_proximity_variants.py` (Breiman, out-of-bag, and RF-GAP proximities), and `run_unknown_k.py` (number of subtypes estimated), with their `analyze_` scripts.
6. **Manuscript assets:** `make_paper_assets.py` generates every table, figure, and reported number.

Each analysis plan is written in the docstring of its run script and was committed before the corresponding results existed. Results are written to `experiments/results/`.

## Project structure

```
btc/                    BTC and trajectory extraction (scikit-learn, XGBoost, LightGBM)
baselines/              Dense and factorized spectral Random Forest Clustering
data/                   Download and preprocessing scripts for the local datasets
experiments/            Study scripts, analyses, and the frozen confirmation protocol
tests/                  Regression tests
```

## License

MIT; see `LICENSE`.
