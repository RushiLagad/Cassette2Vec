# Cassette2Vec-EC

Reading the gene neighborhoods inside genomic islands to classify poultry-associated *Enterococcus cecorum* lineages.

[![Paper](https://img.shields.io/badge/Frontiers_in_Microbiology-10.3389%2Ffmicb.2026.1882753-0a7d74)](https://doi.org/10.3389/fmicb.2026.1882753)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18529389.svg)](https://doi.org/10.5281/zenodo.18529389)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What this is

Most genome classifiers treat a genome as a bag of genes. Cassette2Vec-EC looks at local arrangement instead. Cassettes are contiguous runs of genes inside merged IslandViewer intervals, bounded by contig ends or by non-island gaps. Each cassette carries features built from island context, mobility genes and AMR hits, and models are evaluated with genomes kept together in cross-validation so no genome leaks between train and test.

## Published result

Lagad RR, Rafi S, Goswami A. Genomic-island cassette architecture provides interpretable signal for exploratory classification of poultry-associated *Enterococcus cecorum* lineages. *Frontiers in Microbiology*. 2026;17:1882753. https://doi.org/10.3389/fmicb.2026.1882753

On 145 genomes (95 commensal, 50 pathogenic) under genome-grouped 5-fold cross-validation the paper reports:

| Model | Features | AUROC |
|---|---|---|
| **Cassette2Vec-EC, genome-level (primary)** | 84 | **0.918 ± 0.067** |
| Cassette + GI burden hybrid | 99 | 0.915 ± 0.062 |
| AMR gene content, corrected | 43 | 0.906 ± 0.044 |
| GI-restricted gene presence/absence proxy | 484 | 0.887 ± 0.083 |
| **Cassette-row model (the one archived here)** | 17 | **0.874 ± 0.077** |
| GI burden summaries | 15 | 0.791 ± 0.050 |
| Assembly quality only (contig count) | 1 | 0.743 ± 0.015 |

Leave-one-BioProject-out, training on 20 projects and testing on PRJNA1050746: cassette 0.694, AMR gene content 0.663, GI burden 0.405, contig count 0.314.

The gap between 0.918 and 0.694 is part of the finding. The signal is real inside the training studies and gets weaker on a study the model has never seen, which is why the paper calls the classifier exploratory.

## Which model is in this repository

Read this before comparing anything here against the paper.

The paper's **primary** result is a Random Forest (500 trees, `random_state=42`, balanced class weights) on **84 genome-level** cassette summary features, scoring 0.918.

This repository archives the **v1.1 cassette-row pipeline**: an XGBoost model over **17 per-cassette** features, one row per gene inside a genomic island. The paper reports that model separately at **0.874**. It is a different algorithm on a different feature set at a different level of aggregation, so nothing produced by the scripts here will reproduce 0.918.

Use the paper for reported performance. Use this repository to build cassette features and score new genomes with the archived cassette-row model.

## What is not in this repository

Stated plainly so nobody loses time looking:

- **No training or cross-validation code.** The scripts here build features, add AMR neighbourhood scores and draw figures from a saved predictions file. The genome-grouped CV that produced the published numbers is not included, so the grouping cannot be inspected or rerun from this repository.
- **No locked fold file.** The 145-genome 5-fold assignment used in the paper is not in the current Zenodo record.
- **No accession table.** The commensal and pathogenic accession lists are in Supplementary Data S0a and S0b of the paper, not here.
- **No genome identifiers in the archived predictions.** `cassette2vec_v11_predictions.csv` has `true_label` and `pred_score` only, so per-genome scores cannot be recomputed from it.
- **No Random Forest model.** See the section above.

## Repository layout

```
Cassette2Vec/
├── cassette2vec_predict.py        build cassette features and score new genomes
├── add_AMR_neighborhood_v1.py     add AMR neighborhood features
├── cassette2vec_ML_v11_FINAL.py   evaluation figures and SHAP summary
├── run_pipeline.sh                runs the last two steps in order
├── data/                          put inputs here (see below)
├── scripts/update_readme_zenodo.py   keeps the Zenodo file table below current
├── requirements.txt
├── environment.yml
└── CITATION.cff
```

## Install

```bash
git clone https://github.com/RushiLagad/Cassette2Vec.git
cd Cassette2Vec
pip install -r requirements.txt        # Python 3.11
# or: conda env create -f environment.yml && conda activate cassette2vec
```

## Data

The trained model, feature matrix, predictions and IslandViewer calls are on Zenodo: https://doi.org/10.5281/zenodo.18529389

```bash
pip install zenodo-get
zenodo_get 18529389 -o data/
```

All 145 assemblies are public on NCBI. Accession lists are in Supplementary Data S0a (commensal) and S0b (pathogenic) of the paper.

BioProjects: PRJNA165387, PRJNA191885, PRJNA82533, PRJNA268097, PRJNA284225 to PRJNA284229, PRJNA284445 to PRJNA284446, PRJNA352388, PRJNA377666, PRJNA493784, PRJNA580016, PRJNA704975, PRJNA1050746, PRJDB18532, PRJEB33338, PRJEB57096.

Those 20 BioProjects cover 141 of the 145 genomes. The remaining 4, all pathogenic, were retrieved as direct GCF assembly accessions with no associated BioProject, so rebuilding the cohort from the project list alone will come up short. The accession lists in the supplementary data are the complete record.

## Running it

To build features from raw genomes you need [Prokka 1.14.6](https://github.com/tseemann/prokka), [IslandViewer 4](https://www.pathogenomics.sfu.ca/islandviewer/), [ABRicate 1.0.1](https://github.com/tseemann/abricate) (ResFinder and CARD) and [PIRATE 1.0.4](https://github.com/SionBayliss/PIRATE).

**Score one new genome**

```bash
python cassette2vec_predict.py \
    --genome_id    GCF_NEW_GENOME.1 \
    --islandviewer data/islandviewer/GCF_NEW_GENOME_islands.csv \
    --abricate     data/abricate/GCF_NEW_GENOME_abricate.csv \
    --model        data/cassette2vec_v11_model.pkl \
    --outdir       predictions/
```

Add `--batch` with `--islandviewer_dir` and `--abricate_dir` to score a folder of genomes. Output is a genome-level risk score between 0 and 1, cassette-level scores, and the highest-scoring cassettes for follow-up.

**Add AMR neighborhood features**

```bash
python add_AMR_neighborhood_v1.py \
    --input  data/cassette2vec_ML_features_v1_with_mobility_load.csv \
    --output data/cassette2vec_ML_features_v1_with_AMR_neighborhood.csv
```

**Make the evaluation figures** (ROC, PR, calibration, confusion matrix, score boxplot, SHAP)

```bash
python cassette2vec_ML_v11_FINAL.py \
    --model    data/cassette2vec_v11_model.pkl \
    --preds    data/cassette2vec_v11_predictions.csv \
    --features data/cassette2vec_ML_features_v1_with_AMR_neighborhood.csv \
    --outdir   figures/
```

Or run both of the last two steps with `bash run_pipeline.sh`. Seeds are fixed at 42.

## Features the model uses

| Feature | Meaning |
|---|---|
| `GI_flag` | gene sits inside a predicted genomic island |
| `amr_hit` | ABRicate AMR hit |
| `mobility_score` | integrase, transposase or recombinase marker |
| `integrase_flag`, `transposase_flag`, `recombinase_flag` | the individual mobility markers |
| `start`, `end`, `island_start`, `island_end`, `island_length` | gene and island coordinates |
| `Mobility_Load` | number of mobility genes in the cassette |
| `GI_AMR_density` | share of island genes that are AMR hits |
| `AMR_cluster_size` | AMR genes within 3 genes either side |
| `AMR_neighborhood_score` | mobility context around AMR genes |
| `island_gene_count`, `amr_gene_count_in_island` | island gene and AMR counts |

## What's next

The same approach is being extended to *Enterococcus faecalis*. That work is in progress and not in this repository.

## Citation

Please cite the paper. To cite the code archive as well:

```bibtex
@article{lagad2026cassette,
  author  = {Lagad, Rushikesh R. and Rafi, Shakil and Goswami, Aranyak},
  title   = {Genomic-island cassette architecture provides interpretable signal for exploratory classification of poultry-associated Enterococcus cecorum lineages},
  journal = {Frontiers in Microbiology},
  volume  = {17},
  pages   = {1882753},
  year    = {2026},
  doi     = {10.3389/fmicb.2026.1882753}
}

@software{lagad2026cassette_code,
  author    = {Lagad, Rushikesh R. and Rafi, Shakil and Goswami, Aranyak},
  title     = {Cassette2Vec-EC},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18529389}
}
```

## Authors

- **Rushikesh R. Lagad**: idea, pipeline, genomic analysis, ML experiments, figures
- **Shakil Rafi**: ML methodology and model validation
- **Aranyak Goswami**: supervision and funding. Corresponding author, garanyak@uark.edu

Department of Animal Science and Center for Agricultural Data Analytics, University of Arkansas System Division of Agriculture, Fayetteville, AR.

Funded by the Arkansas Research Alliance Impact Grant Program, with support from Cobb-Vantress Genetics LLC. Computing by the Arkansas High Performance Computing Center.

MIT License. Copyright 2026 Rushikesh R. Lagad, Shakil Rafi, Aranyak Goswami.

<!-- ZENODO_FILES_START -->
> **Auto-generated from Zenodo API** - Last updated: 2026-11-15 08:42 UTC

### 📦 Zenodo Archive

| | |
|---|---|
| **DOI** | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18529389.svg)](https://doi.org/10.5281/zenodo.18529389) |
| **Version** | v1.0.0 |
| **Published** | 2026-02-09 |
| **Files** | 10 files (168.94 MB total) |
| **Views / Downloads** | 95 / 186 |

### 📥 Download Files

| File | Size | Description | Download |
|------|------|-------------|----------|
| `add_AMR_neighborhood_v1.py` | 9.63 KB | Step 2 pipeline - adds AMR neighborhood scores to feature matrix | [⬇ Download](https://zenodo.org/api/records/18529389/files/add_AMR_neighborhood_v1.py/content) |
| `cassette2vec_ML_v11_FINAL.py` | 13.19 KB | Evaluation + SHAP figure generation script | [⬇ Download](https://zenodo.org/api/records/18529389/files/cassette2vec_ML_v11_FINAL.py/content) |
| `cassette2vec_predict.py` | 28.48 KB | Prediction pipeline - run on a new genome | [⬇ Download](https://zenodo.org/api/records/18529389/files/cassette2vec_predict.py/content) |
| `requirements.txt` | 171 B | Pinned Python package versions | [⬇ Download](https://zenodo.org/api/records/18529389/files/requirements.txt/content) |
| `environment.yml` | 328 B | Conda environment specification | [⬇ Download](https://zenodo.org/api/records/18529389/files/environment.yml/content) |
| `RushiLagad/Cassette2Vec-v1.0.0.zip` | 12.74 KB | - | [⬇ Download](https://zenodo.org/api/records/18529389/files/RushiLagad/Cassette2Vec-v1.0.0.zip/content) |
| `cassette2vec_ML_features_v1_with_mobility_load.csv` | 166.09 MB | Base cassette feature matrix (145 genomes, 51,302 rows) | [⬇ Download](https://zenodo.org/api/records/18529389/files/cassette2vec_ML_features_v1_with_mobility_load.csv/content) |
| `cassette2vec_islandviewer_all_clean.csv` | 1.58 MB | IslandViewer genomic island calls (all 145 genomes, cleaned) | [⬇ Download](https://zenodo.org/api/records/18529389/files/cassette2vec_islandviewer_all_clean.csv/content) |
| `cassette2vec_v11_predictions.csv` | 754.96 KB | Cassette-level model predictions (`true_label`, `pred_score`) | [⬇ Download](https://zenodo.org/api/records/18529389/files/cassette2vec_v11_predictions.csv/content) |
| `cassette2vec_v11_model.pkl` | 481.70 KB | Trained XGBoost model - load with `joblib.load()` | [⬇ Download](https://zenodo.org/api/records/18529389/files/cassette2vec_v11_model.pkl/content) |

### 🚀 Quick Download (command line)

```bash
pip install zenodo-get
zenodo_get 18529389 -o data/
```
<!-- ZENODO_FILES_END -->



