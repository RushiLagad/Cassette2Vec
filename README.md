# Cassette2Vec
Machine learning framework for Enterococcus cecorum pathogenicity prediction (Antimicrobial Resistance + genomic islands + pangenome)
# CASSETTE2VEC v1  Machine Learning Model for *Enterococcus cecorum* Pathogenicity Prediction

CASSETTE2VEC is a genomic cassette–aware machine learning framework designed to predict **pathogenic vs. commensal** *Enterococcus cecorum* genomes using antimicrobial resistance (AMR) signatures, genomic islands, pangenome structure, and genome-level features.

This repository contains (or will contain):

- Reproducible feature engineering pipeline  
- Cleaned master datasets (schema and examples)  
- Machine learning model training workflows  
- Prediction scripts for new genomes  
- Documentation for integration with wet-lab sequencing workflows  

---

## ✨ Key Features (v1)

- AMR gene counts (total and class-wise) from ABRicate  
- Virulence gene counts  
- Genomic island metrics (GI count, total GI length, GI gene count) from IslandViewer  
- Pangenome presence–absence matrix from PIRATE  
- Genome statistics (genome size, GC%) from Prokka annotations  
- Baseline models: Logistic Regression, Random Forest, XGBoost  
- Evaluation: ROC–AUC, PR–AUC, calibration, confusion matrix  
- Interpretable feature importance for biological insight  

---

## 📂 Planned Repository Structure

```text
cassette2vec/
├── features/
├── preprocessing/
├── models/
├── utils/
└── cli/

data/
├── metadata/
├── abricate/
├── islandviewer/
├── pirate/
├── eggnog/
└── master_tables/

notebooks/
├── 01_feature_engineering.ipynb
├── 02_train_models.ipynb
├── 03_evaluation.ipynb
└── 04_real_time_prediction.ipynb


<!-- ZENODO_FILES_START -->
> **Auto-generated from Zenodo API** — Last updated: 2026-03-03 00:54 UTC

### 📦 Zenodo Archive

| | |
|---|---|
| **DOI** | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18529389.svg)](https://doi.org/10.5281/zenodo.18529389) |
| **Version** | v1.0.0 |
| **Published** | 2026-02-09 |
| **Files** | 10 files (168.94 MB total) |
| **Views / Downloads** | 8 / 0 |

### 📥 Download Files

| File | Size | Description | Download |
|------|------|-------------|----------|
| `add_AMR_neighborhood_v1.py` | 9.63 KB | Step 2 pipeline — adds AMR neighborhood scores to feature matrix | [⬇ Download](https://zenodo.org/api/records/18529389/files/add_AMR_neighborhood_v1.py/content) |
| `cassette2vec_ML_v11_FINAL.py` | 13.19 KB | Evaluation + SHAP figure generation script | [⬇ Download](https://zenodo.org/api/records/18529389/files/cassette2vec_ML_v11_FINAL.py/content) |
| `cassette2vec_predict.py` | 28.48 KB | Prediction pipeline — run on a new genome | [⬇ Download](https://zenodo.org/api/records/18529389/files/cassette2vec_predict.py/content) |
| `requirements.txt` | 171 B | Pinned Python package versions | [⬇ Download](https://zenodo.org/api/records/18529389/files/requirements.txt/content) |
| `environment.yml` | 328 B | Conda environment specification | [⬇ Download](https://zenodo.org/api/records/18529389/files/environment.yml/content) |
| `RushiLagad/Cassette2Vec-v1.0.0.zip` | 12.74 KB | — | [⬇ Download](https://zenodo.org/api/records/18529389/files/RushiLagad/Cassette2Vec-v1.0.0.zip/content) |
| `cassette2vec_ML_features_v1_with_mobility_load.csv` | 166.09 MB | Base cassette feature matrix (145 genomes, 51,302 rows) | [⬇ Download](https://zenodo.org/api/records/18529389/files/cassette2vec_ML_features_v1_with_mobility_load.csv/content) |
| `cassette2vec_islandviewer_all_clean.csv` | 1.58 MB | IslandViewer genomic island calls (all 145 genomes, cleaned) | [⬇ Download](https://zenodo.org/api/records/18529389/files/cassette2vec_islandviewer_all_clean.csv/content) |
| `cassette2vec_v11_predictions.csv` | 754.96 KB | Cassette-level model predictions (`true_label`, `pred_score`) | [⬇ Download](https://zenodo.org/api/records/18529389/files/cassette2vec_v11_predictions.csv/content) |
| `cassette2vec_v11_model.pkl` | 481.70 KB | Trained XGBoost model — load with `joblib.load()` | [⬇ Download](https://zenodo.org/api/records/18529389/files/cassette2vec_v11_model.pkl/content) |

### 🚀 Quick Download (command line)

```bash
pip install zenodo-get
zenodo_get 18529389 -o data/
```
<!-- ZENODO_FILES_END -->

