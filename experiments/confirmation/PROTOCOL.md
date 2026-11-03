# Confirmation Protocol: What in a Boosting History Reveals Subtypes?

Status: FROZEN at the commit that adds this version. The confirmation run must start from that commit, and the code it uses must not change before the run completes.

## Purpose

Development work on 13 benchmarks and OpenML-CC18 shaped every choice below. This protocol tests the resulting claims once, on 40 datasets that no method has been run on.

## Data

- **Confirmation list:** `confirmation_datasets.csv`, 40 OpenML datasets. They were produced by `build_confirmation.py`, which selects candidates from metadata, applies mechanical validity rules, and then applies two documented review lists. `confirmation_audit.csv` records the reason for each of the 61 exclusions.
- **Validity rules:** no class may be separable by a single feature (AUC of at least 0.999), at most 20% duplicated rows, at most 50% missing values, at least 4 features after encoding, and at least 3 classes with 5 or more members. Ordinal targets, duplicates of other datasets, and overlapping classes were excluded by review of names and class levels only.
- **Size:** datasets larger than 20,000 rows are subsampled to 20,000 rows with a fixed seed. Rows with a missing target are dropped.
- **Preprocessing:** `clean_features` in `experiments/dataset_loader.py`, unchanged for all datasets.

## Proxy tasks

Each dataset receives three random class groupings with a fixed seed. Each grouping puts at least one class in the background (y=0) and at least two classes in the target cohort (y=1), whose classes are the subtypes. Each grouping runs 5 seeds. Results average to one value per dataset, which is the unit of analysis.

Random groupings are not meaningful coarse labels. The confirmation therefore tests how representations behave, not whether real clinical proxies help.

## Methods (frozen in `experiments/run_confirmation.py`)

- **BTC:** 100 depth-1 stumps, learning rate 0.1, square-root feature subsampling, per-tree updates, K-Means with 10 restarts.
- **TreeSHAP:** XGBoost with depth 1, 100 trees, learning rate 0.1, column subsampling 0.8, SHAP values standardized and clustered with K-Means. This was the best of 24 SHAP setups in development, a choice that favors SHAP.
- **Per-feature sums:** BTC updates summed by split feature.
- **History representations:** pseudo-residuals, pseudo-residuals with tree order shuffled, and the final margin.
- **RF leaves with SVD:** 100-tree random forest, one-hot leaves, up to 50 SVD components, K-Means.
- **Unsupervised:** K-Means on raw and on standardized features, and UMAP on raw and on standardized features, each followed by K-Means. UMAP runs only for target cohorts of at most 20,000 samples.
- Every method receives the true number of subtypes.

## Claims and how they are judged

Each comparison is the mean ARI difference across datasets with a 95% dataset-level bootstrap interval from 20,000 resamples. A claim of difference requires the interval to exclude zero. Results are reported in full whether or not a claim is made, and no equivalence verdict is used.

| Claim | Comparison | Status |
|---|---|---|
| C1. Keeping per-tree updates beats SHAP clustering | BTC minus TreeSHAP | Primary |
| C1, supporting | BTC minus per-feature sums | Secondary |
| C2. Learning history adds little beyond the final score | Pseudo-residuals minus final margin | Secondary |
| C2, supporting | Pseudo-residuals minus shuffled-order pseudo-residuals | Secondary |
| C3. Comparison with other representations | BTC minus RF leaves with SVD, both K-Means baselines, and both UMAP baselines | Descriptive |
| C4. Robustness to scaling | Mean absolute difference between raw and standardized K-Means, and the share of datasets where it exceeds 0.05. BTC is exactly invariant to feature scaling | Descriptive |
| E1. Leaf homogeneity as a diagnostic | Rank correlation between homogeneity and BTC ARI across groupings | Exploratory, since a fixed-cohort development test found no relationship |

Secondary claims are judged with the same interval rule but are labelled secondary in the paper. Running time is recorded during parallel execution and is reported descriptively only.

## Development evidence behind the choices

| Comparison on the 13 benchmarks | Difference | 95% interval |
|---|---|---|
| BTC minus best of 24 TreeSHAP setups | +0.066 | 0.018 to 0.125 |
| BTC minus RF leaves with SVD | +0.018 | -0.034 to 0.080 |
| BTC minus standardized K-Means | +0.025 | -0.108 to 0.159 |
| Pseudo-residuals minus final margin | +0.019 | -0.004 to 0.044 |

On 27 CC18 datasets with random groupings, RF leaves with SVD scored above BTC (interval for BTC minus RF -0.107 to -0.018), and raw K-Means scored above both. These results are why the protocol makes no claim that BTC beats RF leaves or unsupervised clustering.

## Known limitations

- Random groupings stand in for real coarse labels.
- Some exclusions required judgement about ordinal targets, made from names and class levels before any results.
- The loader and dataset descriptions were corrected after development results were seen. The corrections restore standard preprocessing and are listed in the git history.
- With 40 datasets and a development standard deviation of about 0.06 to 0.12 for these differences, intervals will be roughly 0.02 to 0.04 wide on each side.

## Rules after the run

The confirmation runs once. Any bug fix that changes results must be disclosed with both sets of results. Analyses not listed here are labelled exploratory.

Code commit for the confirmation run: e2dc623 on branch data-audit-fixes of the code repository.
