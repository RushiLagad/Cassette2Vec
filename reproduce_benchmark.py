#!/usr/bin/env python3
"""
reproduce_benchmark.py

Reruns the primary benchmark of Lagad, Rafi and Goswami (2026), Frontiers in
Microbiology 17:1882753: a Random Forest on the 84 genome-level cassette summary
features, evaluated with the locked genome-grouped 5-fold assignment and with
leave-one-BioProject-out.

Provenance. This script was written after publication from the protocol stated
in the paper's Methods (500 trees, random_state 42, balanced class weights,
n_jobs 1, the locked fold file). It is not the script that produced the
published tables. Run under scikit-learn 1.3.2 it reproduces the per-fold
AUROC, AUPRC and Brier values in Supplementary Data S2 to three decimals
(one AUPRC value differs by 0.001), the means of 0.918 and 0.906, and the
leave-one-BioProject-out AUROC of 0.694. Under newer scikit-learn releases
the Random Forest draws differ and the mean AUROC shifts, for example to
0.914 under 1.9.1, so use the pinned environment.

Inputs (in data/benchmark/ of this repository, .csv or .xlsx accepted):
  cassette_genome_summary_matrix_145genomes.csv   145 genomes x 84 features
  145genome_cv5_fold_file.csv                     locked folds, labels, BioProjects

Usage:
  python reproduce_benchmark.py \
      --matrix data/benchmark/cassette_genome_summary_matrix_145genomes.csv \
      --folds  data/benchmark/145genome_cv5_fold_file.csv
"""
import argparse

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ID_COLS = ["genome_id", "genome_class", "is_pathogenic"]


def read_table(path):
    """Read .csv or .xlsx by extension."""
    return pd.read_csv(path) if str(path).lower().endswith(".csv") else pd.read_excel(path)
HELD_OUT_PROJECT = "PRJNA1050746"


def make_model():
    return RandomForestClassifier(
        n_estimators=500, random_state=42, class_weight="balanced", n_jobs=1
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--matrix", required=True, help="84-feature cassette summary matrix (.csv or .xlsx)")
    ap.add_argument("--folds", required=True, help="locked 145-genome fold file (.csv or .xlsx)")
    args = ap.parse_args()

    mat = read_table(args.matrix)
    folds = read_table(args.folds)[["genome_id", "bioproj", "cv5_fold_id"]]
    df = mat.merge(folds, on="genome_id", how="inner", validate="one_to_one")

    features = [c for c in mat.columns if c not in ID_COLS]
    X = df[features].fillna(0).to_numpy(dtype=float)
    y = df["is_pathogenic"].to_numpy(dtype=int)
    fold = df["cv5_fold_id"].to_numpy(dtype=int)
    project = df["bioproj"].astype(str).to_numpy()

    print(f"scikit-learn {sklearn.__version__}")
    print(f"{len(df)} genomes, {len(features)} features, "
          f"{int(y.sum())} pathogenic, {int((1 - y).sum())} commensal\n")

    print("Genome-grouped 5-fold cross-validation (locked folds)")
    print("fold    n   AUROC   AUPRC   Brier")
    auroc, auprc, brier = [], [], []
    for k in sorted(np.unique(fold)):
        test = fold == k
        model = make_model().fit(X[~test], y[~test])
        p = model.predict_proba(X[test])[:, 1]
        auroc.append(roc_auc_score(y[test], p))
        auprc.append(average_precision_score(y[test], p))
        brier.append(brier_score_loss(y[test], p))
        print(f"  {k}   {int(test.sum()):3d}   {auroc[-1]:.3f}   {auprc[-1]:.3f}   {brier[-1]:.3f}")
    print(f"mean        {np.mean(auroc):.3f}   {np.mean(auprc):.3f}   {np.mean(brier):.3f}")
    print(f"SD (n-1)    {np.std(auroc, ddof=1):.3f}   {np.std(auprc, ddof=1):.3f}    as in Supplementary Data S2")
    print(f"SD (n)      {np.std(auroc, ddof=0):.3f}   {np.std(auprc, ddof=0):.3f}    as in Table 1 of the paper\n")

    test = project == HELD_OUT_PROJECT
    model = make_model().fit(X[~test], y[~test])
    p = model.predict_proba(X[test])[:, 1]
    print(f"Leave-one-BioProject-out, testing on {HELD_OUT_PROJECT}")
    print(f"train n = {int((~test).sum())}, test n = {int(test.sum())} "
          f"({int(y[test].sum())} pathogenic, {int((1 - y[test]).sum())} commensal)")
    print(f"AUROC {roc_auc_score(y[test], p):.3f}")


if __name__ == "__main__":
    main()
