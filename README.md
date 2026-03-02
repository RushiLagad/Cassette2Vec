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
