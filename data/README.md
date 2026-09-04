# Benchmark Datasets and Provenance

This directory contains the preprocessing scripts and data management logic for the 13 benchmark datasets evaluated in **Boosting Trajectory Clustering (BTC)**.

To adhere to open-source repository size limits and ensure complete reproducibility from original sources, raw data files (`.pkl`, `.gz`, `.h5`) are not tracked in Git. Instead, the repository provides automated, deterministic scripts to download and preprocess all datasets directly from official repositories.

---

## Quick Setup: Downloading All Data

To automatically download and prepare all biomedical datasets from scratch, run:

```bash
uv run python download_all_data.py
```

This single command fetches and constructs all local datasets. Standard tabular benchmarks (e.g., Dermatology, Vehicle, Thyroid, HAR) are fetched dynamically via `openml` during benchmark runtime.

---

## Detailed Dataset Provenance & Sources

| Dataset | Domain | Source / Accession | Proxy Task ($y=0$ vs. $y=1$) | Target Subtypes ($K$) | Paradigm |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Diabetes** | Clinical Tabular | UCI / OpenML ID 4541 (Strack et al., 2014) | No Readmission vs. Readmitted | 4 ICD-9 Categories: Circulatory, Respiratory, Digestive, Diabetes | Hierarchical |
| **BRCA** | Cancer Genomics | TCGA / UCSC Xena Hub (`TCGA.BRCA.sampleMap`) | ER-Negative vs. ER-Positive | 4 PAM50 Subtypes: Basal, HER2, LumA, LumB | Cross-Cutting |
| **GBMLGG** | Cancer Genomics | TCGA / UCSC Xena Hub (`TCGA.GBMLGG.sampleMap`) | Alive at Follow-up vs. Deceased | 3 Histological Grades: Grade II, Grade III, Grade IV (GBM) | Cross-Cutting |
| **Paul15** | Single-cell RNA-seq | Paul et al. (2015) via `scanpy.datasets.paul15()` | Non-Myeloid vs. Myeloid Progenitors | 6 Myeloid Differentiation Stages | Hierarchical |
| **CTG** | Cardiotocography | UCI / OpenML ID 1560 | Normal State vs. Abnormal State | 2 Pathology Subtypes: Suspect vs. Pathologic | Hierarchical |
| **Mice** | Proteomics | UCI / OpenML ID 40966 (Higuera et al., 2015) | Control Genotype vs. Trisomic (Ts65Dn) | 4 Genotype/Treatment Combinations | Hierarchical |
| **Thyroid** | Clinical Tabular | UCI / OpenML ID 57 | Normal Thyroid vs. Abnormal Function | 3 Thyroid Pathology Subtypes | Hierarchical |
| **HAR** | Motion Sensors | UCI / OpenML ID 1478 | Static (sit/stand/lie) vs. Dynamic Activities | 3 Dynamic Walking Subtypes | Hierarchical |
| **Dermatology** | Clinical Tabular | UCI / OpenML ID 35 | Psoriasis (Majority) vs. Other Conditions | 5 Dermatological Disease Subtypes | Hierarchical |
| **Vehicle** | Silhouette Vision | StatLib / OpenML ID 54 | Opel Silhouette vs. Other Silhouettes | 3 Vehicle Silhouette Types (Saab, Bus, Van) | Hierarchical |
| **Segment** | Image Segmentation | StatLib / OpenML ID 36 | Brickface Segment vs. Other Segments | 6 Image Subtypes (Sky, Foliage, Cement, Window, Path, Grass) | Hierarchical |
| **Arrhythmia** | Cardiac ECG | UCI / OpenML ID 5 | Normal ECG vs. Cardiac Arrhythmia | 12 Arrhythmia Subtypes | Hierarchical |
| **Leukemia** | Microarray Genomics | Golub et al. (1999) / OpenML ID 45092 | AML (Non-lymphoblastic) vs. ALL | 2 Subtypes: B-cell ALL vs. T-cell ALL | Hierarchical |

---

## Clinical Derivation of the Diabetes Dataset

The Diabetes benchmark is derived from the 10-year clinical database of hospitalized diabetic patients across 130 US hospitals (Strack et al., 2014):
* **Source:** OpenML ID 4541 (`Diabetes130US`).
* **Encounter Sample:** A deterministic cohort of $N = 20{,}000$ patient encounters characterized by 22 clinical, demographic, and medication features.
* **Proxy Task:** Binary hospital readmission indicator ($y=1$ if readmitted within or after 30 days; $y=0$ if not readmitted).
* **Target Subtypes ($K=4$):** Evaluated strictly on the readmitted patient cohort ($y=1$), mapped using primary ICD-9-CM diagnosis codes into the 4 dominant clinical disease categories:
  1. *Circulatory Diseases* (ICD-9 `390–459`, `785`): Congestive heart failure, coronary atherosclerosis, acute myocardial infarction, hypertension.
  2. *Respiratory Diseases* (ICD-9 `460–519`, `786`): Pneumonia, chronic bronchitis, respiratory failure.
  3. *Digestive Diseases* (ICD-9 `520–579`, `787`): Gastroenteritis, ulcers, diabetic gastroparesis.
  4. *Diabetes Complications* (ICD-9 `250.xx`): Acute glycemic crises (ketoacidosis, hyperosmolar coma, uncontrolled hyperglycemia).
* **Exclusions:** The generic "Other" catch-all and non-metabolic incidental admissions (accidental trauma/injury, neoplasms, musculoskeletal disorders) were excluded to maintain intra-cluster semantic coherence and avoid biasing clustering evaluation.

---

## Directory Organization

```
data/
├── README.md                     # This documentation file
├── ctg/
│   └── preprocess_ctg.py         # OpenML CTG downloader and preprocessor
├── diabetes/
│   └── preprocess_diabetes.py    # OpenML Diabetes130US downloader and ICD-9 preprocessor
├── mice/
│   └── preprocess_mice.py        # OpenML Mice downloader and imputer
├── scrna/
│   └── preprocess_scrna.py       # Paul15 scanpy downloader and preprocessor
├── tcga/
│   ├── preprocess_brca.py        # UCSC Xena TCGA BRCA downloader and preprocessor
│   └── update_brca_staging.py    # GDC clinical staging annotator
├── tcga_gbmlgg/
│   └── preprocess_gbmlgg.py      # UCSC Xena TCGA GBMLGG downloader and preprocessor
└── thyroid/                      # Fetched on-the-fly via dataset_loader.py
```
