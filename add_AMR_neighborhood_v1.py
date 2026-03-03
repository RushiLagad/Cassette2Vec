#!/usr/bin/env python3
"""
add_AMR_neighborhood_v1.py
==========================
Add AMR_neighborhood_score and AMR_cluster_size to the Cassette2Vec feature table.

This is Step 2 of the Cassette2Vec-EC pipeline — must be run BEFORE
cassette2vec_ML_v11_FINAL.py (which needs the output file for SHAP plots).

Pipeline order:
  STEP 1: cassette2vec_predict.py    — builds base feature matrix
  STEP 2: add_AMR_neighborhood_v1.py — adds AMR neighborhood features  ← THIS SCRIPT
  STEP 3: cassette2vec_ML_v11_FINAL.py — evaluation plots + SHAP

Scoring logic (applied only to AMR genes where amr_hit == 1):
  For each AMR gene, examine a ±3-gene window (up to 7 genes total)
  within the same (genome_id, contig, GI_ID) group, sorted by genomic start.

  Scoring:
    Any integrase_flag  == 1 in window  →  +2
    Any transposase_flag == 1 in window →  +2
    Any recombinase_flag == 1 in window →  +1
    AMR cluster size in window >= 2     →  +1

  Outputs per gene:
    AMR_neighborhood_score  (float, 0–6)
    AMR_cluster_size        (int, count of AMR genes in window)

  Non-AMR genes (amr_hit == 0) receive score = 0, cluster_size = 0.

Usage:
  # Default: reads/writes from data/ folder relative to repo root
  python add_AMR_neighborhood_v1.py

  # Custom paths:
  python add_AMR_neighborhood_v1.py \\
      --input  data/cassette2vec_ML_features_v1_with_mobility_load.csv \\
      --output data/cassette2vec_ML_features_v1_with_AMR_neighborhood.csv

Repository : https://github.com/RushiLagad/Cassette2Vec (tag v1.1.0)
Zenodo DOI : https://doi.org/10.5281/zenodo.18529389
"""

# ── Reproducibility seeds ─────────────────────────────────────────────────
import random
import numpy as np
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Standard imports ──────────────────────────────────────────────────────
import argparse
import sys
from pathlib import Path

import pandas as pd


# ══════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════

def parse_args():
    repo_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Add AMR_neighborhood_score and AMR_cluster_size to Cassette2Vec feature table",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=repo_root / "data" / "cassette2vec_ML_features_v1_with_mobility_load.csv",
        help="Input feature CSV (default: data/cassette2vec_ML_features_v1_with_mobility_load.csv)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=repo_root / "data" / "cassette2vec_ML_features_v1_with_AMR_neighborhood.csv",
        help="Output feature CSV (default: data/cassette2vec_ML_features_v1_with_AMR_neighborhood.csv)",
    )
    parser.add_argument(
        "--window", "-w",
        type=int,
        default=3,
        help="Half-window size for neighborhood scoring (default: 3, i.e. ±3 genes)",
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════
# MAIN LOGIC
# ══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Validate input ────────────────────────────────────────────────────
    if not args.input.exists():
        sys.exit(
            f"ERROR: Input file not found: {args.input}\n"
            "       Generate this file first by running cassette2vec_predict.py\n"
            "       or download it from Zenodo: https://doi.org/10.5281/zenodo.18529389"
        )

    # ── Load feature table ────────────────────────────────────────────────
    print(f"[INFO] Loading feature table from: {args.input}")
    df = pd.read_csv(args.input, low_memory=False)
    print(f"[INFO] Loaded {len(df):,} rows × {len(df.columns)} columns")

    # ── Validate required columns ─────────────────────────────────────────
    required_cols = [
        "genome_id", "contig", "GI_ID", "start",
        "amr_hit", "integrase_flag", "transposase_flag", "recombinase_flag",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        sys.exit(
            f"ERROR: Missing required columns: {missing}\n"
            f"       Found columns: {df.columns.tolist()}"
        )

    # ── Check if already processed ────────────────────────────────────────
    if "AMR_neighborhood_score" in df.columns:
        print("[WARN] Column 'AMR_neighborhood_score' already exists — overwriting.")

    # ── Ensure flag columns are numeric 0/1 ──────────────────────────────
    flag_cols = ["amr_hit", "integrase_flag", "transposase_flag", "recombinase_flag"]
    for col in flag_cols:
        df[col] = df[col].fillna(0).astype(int)

    n_amr_genes = df["amr_hit"].sum()
    print(f"[INFO] AMR genes (amr_hit=1): {n_amr_genes:,} / {len(df):,} total genes")
    if n_amr_genes == 0:
        print("[WARN] No AMR genes found. All scores will be 0.")

    # ── Sort for genomic neighborhood order ──────────────────────────────
    df = df.sort_values(["genome_id", "contig", "GI_ID", "start"]).reset_index(drop=True)

    # ── Initialise output columns ─────────────────────────────────────────
    df["AMR_neighborhood_score"] = 0.0
    df["AMR_cluster_size"]       = 0

    # ── Compute scores ────────────────────────────────────────────────────
    print(f"[INFO] Computing AMR neighborhood scores (±{args.window} gene window)...")

    group_cols   = ["genome_id", "contig", "GI_ID"]
    n_groups     = df.groupby(group_cols).ngroups
    n_scored     = 0

    for g_idx, (_, group) in enumerate(df.groupby(group_cols)):
        if g_idx % 500 == 0:
            print(f"  ... {g_idx:,} / {n_groups:,} groups processed", end="\r")

        idx_list = list(group.index)

        for pos, idx in enumerate(idx_list):
            if df.at[idx, "amr_hit"] != 1:
                continue  # only score AMR genes

            # ±window genes (slice is exclusive on right)
            lo         = max(0, pos - args.window)
            hi         = min(len(idx_list), pos + args.window + 1)
            window_idx = idx_list[lo:hi]
            window     = df.loc[window_idx]

            has_int      = (window["integrase_flag"]   == 1).any()
            has_trn      = (window["transposase_flag"]  == 1).any()
            has_rec      = (window["recombinase_flag"]  == 1).any()
            cluster_size = int(window["amr_hit"].sum())

            score = 0.0
            if has_int:           score += 2.0
            if has_trn:           score += 2.0
            if has_rec:           score += 1.0
            if cluster_size >= 2: score += 1.0

            df.at[idx, "AMR_neighborhood_score"] = score
            df.at[idx, "AMR_cluster_size"]       = cluster_size
            n_scored += 1

    print(f"\n[INFO] Scored {n_scored:,} AMR genes across {n_groups:,} GI groups")

    # ── Summary statistics (AMR genes only) ───────────────────────────────
    amr_mask = df["amr_hit"] == 1
    if amr_mask.any():
        print("\n[INFO] AMR_neighborhood_score summary (AMR genes only):")
        print(df.loc[amr_mask, "AMR_neighborhood_score"].describe().round(3).to_string())
        print("\n[INFO] AMR_cluster_size summary (AMR genes only):")
        print(df.loc[amr_mask, "AMR_cluster_size"].describe().round(3).to_string())

        score_dist = df.loc[amr_mask, "AMR_neighborhood_score"].value_counts().sort_index()
        print("\n[INFO] AMR_neighborhood_score distribution:")
        for score_val, count in score_dist.items():
            print(f"  score {score_val:.0f} : {count:,} genes")

    # ── Save output ───────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\n[SAVED] → {args.output}")
    print(f"[INFO]  Rows: {len(df):,}  |  Columns: {len(df.columns)}")
    print("\n[DONE] Next step: run cassette2vec_ML_v11_FINAL.py for SHAP plots\n")


if __name__ == "__main__":
    main()
