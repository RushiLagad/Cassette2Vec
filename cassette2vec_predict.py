#!/usr/bin/env python3
"""
cassette2vec_predict.py
=======================
Cassette2Vec-EC v1.1 — Full Prediction Pipeline for New Genomes

Given a new Enterococcus cecorum genome, this script:
  Step 1 — Loads IslandViewer CSV output for the genome
  Step 2 — Loads ABRicate AMR screening output for the genome
  Step 3 — Builds cassette-level feature vectors
  Step 4 — Computes AMR neighborhood scores
  Step 5 — Runs the trained XGBoost model
  Step 6 — Aggregates cassette-level scores to a genome-level risk score
  Step 7 — Prints and saves the prediction

Usage:
  python cassette2vec_predict.py \\
      --genome_id   GCF_000379745.1 \\
      --islandviewer data/islandviewer/GCF_000379745.1_islands.csv \\
      --abricate     data/abricate/GCF_000379745.1_abricate.csv \\
      --model        data/cassette2vec_v11_model.pkl \\
      --outdir       predictions/

  # Batch mode — run on a folder of IslandViewer + ABRicate files:
  python cassette2vec_predict.py --batch \\
      --islandviewer_dir  data/islandviewer/ \\
      --abricate_dir      data/abricate/ \\
      --model             data/cassette2vec_v11_model.pkl \\
      --outdir            predictions/

Input file formats
------------------
IslandViewer CSV (one row per gene in a predicted island):
  Columns: Island start, Island end, Length, Method, Gene name, Gene ID,
           Locus, Gene start, Gene end, Strand, Product, External Annotations, genome

ABRicate merged CSV (standard ABRicate output):
  Columns: sample, SEQUENCE, START, END, STRAND, GENE, PERCENT_COVERAGE,
           PERCENT_IDENTITY, PRODUCT  (plus other standard ABRicate columns)

Notes
-----
- IslandViewer genome IDs use no underscores/dots (e.g. GCF0003797451).
  This script normalises both to a common format automatically.
- ABRicate hit thresholds applied: PERCENT_IDENTITY >= 90, PERCENT_COVERAGE >= 80
  (matching the thresholds observed in the training data).
- The 17 active model features are: GI_flag, amr_hit, mobility_score,
  start, end, island_start, island_end, integrase_flag, transposase_flag,
  recombinase_flag, GI_AMR_density, AMR_cluster_size, AMR_neighborhood_score,
  Mobility_Load, GI_length, island_gene_count, amr_gene_count_in_island
- Three reserved columns (eggNOG_class, pathway, cluster_id) are NOT used.

Repository : https://github.com/RushiLagad/Cassette2Vec (tag v1.1.0)
Zenodo DOI : https://doi.org/10.5281/zenodo.18529389
"""

# ── Reproducibility seeds ──────────────────────────────────────────────────
import random, numpy as np
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Standard imports ───────────────────────────────────────────────────────
import argparse, sys, warnings
from pathlib import Path

import pandas as pd
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

# ── Constants ──────────────────────────────────────────────────────────────
# ABRicate quality thresholds (matching training data)
MIN_IDENTITY = 90.0
MIN_COVERAGE = 80.0

# Minimum island size in bp (matching Methods: ≥5 kb filter)
MIN_ISLAND_BP = 5000

# Mobility keyword patterns (from Prokka product field)
INTEGRASE_KW   = ["integrase"]
TRANSPOSASE_KW = ["transposase", "is200", "is605", "is6 ", "is256", "is1595",
                  "is1297", "is110", "insertion sequence"]
RECOMBINASE_KW = ["recombinase", "xerc", "xerd", "pinr", "site-specific recombinase"]

# The 17 active features the model was trained on (exact order matters)
MODEL_FEATURES = [
    "GI_flag",
    "amr_hit",
    "mobility_score",
    "start",
    "end",
    "island_start",
    "island_end",
    "integrase_flag",
    "transposase_flag",
    "recombinase_flag",
    "GI_AMR_density",
    "AMR_cluster_size",
    "AMR_neighborhood_score",
    "Mobility_Load",
    "island_length",
    "island_gene_count",
    "amr_gene_count_in_island",
]


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def normalise_genome_id(gid: str) -> str:
    """Normalise genome ID to canonical GCF_XXXXXXXXX.X format.
    
    Handles:
      GCF0003797451       → GCF_000379745.1   (IslandViewer strips underscores/dots)
      GCF_000379745.1     → GCF_000379745.1   (ABRicate standard)
      GCF_000379745_1     → GCF_000379745.1
    """
    gid = str(gid).strip()
    if "_" in gid:
        # Already has underscores — just ensure dot-version
        parts = gid.replace(".", "_").split("_")
        if len(parts) >= 3:
            accession = "_".join(parts[:3])
            version = parts[3] if len(parts) > 3 else "1"
            return f"{accession}.{version}"
        return gid
    else:
        # IslandViewer format: GCF0003797451 → GCF_000379745.1
        digits = gid.replace("GCF", "").replace("GCA", "")
        prefix = "GCF" if "GCF" in gid.upper() else "GCA"
        if len(digits) >= 10:
            return f"{prefix}_{digits[:9]}.{digits[9:]}"
        return gid


def flag_mobility(product: str) -> tuple:
    """Return (integrase_flag, transposase_flag, recombinase_flag) for a product string."""
    if not isinstance(product, str):
        return 0, 0, 0
    p = product.lower()
    integrase  = int(any(kw in p for kw in INTEGRASE_KW))
    transposase = int(any(kw in p for kw in TRANSPOSASE_KW))
    recombinase = int(any(kw in p for kw in RECOMBINASE_KW))
    return integrase, transposase, recombinase


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD AND VALIDATE ISLANDVIEWER OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def load_islandviewer(path: Path, genome_id: str) -> pd.DataFrame:
    """
    Load IslandViewer CSV for one genome.
    
    Expected columns:
      Island start, Island end, Length, Method,
      Gene name, Gene ID, Locus, Gene start, Gene end,
      Strand, Product, External Annotations, genome
    
    Returns a cleaned DataFrame with standardised column names.
    """
    print(f"  [IV] Loading IslandViewer: {path.name}")
    df = pd.read_csv(path, low_memory=False)

    # Normalise column names
    df.columns = df.columns.str.strip()
    rename = {
        "Island start": "island_start",
        "Island end":   "island_end",
        "Length":       "island_length",
        "Method":       "iv_method",
        "Gene name":    "gene_name",
        "Gene ID":      "gene_id",
        "Locus":        "locus_tag",
        "Gene start":   "start",
        "Gene end":     "end",
        "Strand":       "strand",
        "Product":      "product",
        "External Annotations": "ext_annotations",
        "genome":       "genome_raw",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # ── Filter: ≥5 kb islands only ────────────────────────────────────────
    before = len(df)
    df = df[df["island_length"] >= MIN_ISLAND_BP].copy()
    print(f"  [IV] Kept {len(df):,} / {before:,} genes after ≥{MIN_ISLAND_BP//1000} kb filter")

    if len(df) == 0:
        print(f"  [IV] WARNING: No islands ≥{MIN_ISLAND_BP} bp found for {genome_id}. "
              "This genome may have very few/small islands.")
        return df

    # ── Normalise genome ID ───────────────────────────────────────────────
    df["genome_id"] = normalise_genome_id(genome_id)

    # ── Merge overlapping GI intervals (union merge) ──────────────────────
    # Sort by contig (all on one genome so island coords are genome-wide)
    df = df.sort_values(["island_start", "start"]).reset_index(drop=True)

    # Assign a merged GI group ID per island interval
    gi_intervals = df[["island_start", "island_end"]].drop_duplicates().sort_values("island_start")
    merged = []
    current_start, current_end, gi_id = None, None, 0
    for _, row in gi_intervals.iterrows():
        s, e = int(row["island_start"]), int(row["island_end"])
        if current_start is None:
            current_start, current_end, gi_id = s, e, 1
        elif s <= current_end:                         # overlapping — extend
            current_end = max(current_end, e)
        else:                                           # gap — new island
            merged.append((current_start, current_end, gi_id))
            gi_id += 1
            current_start, current_end = s, e
    if current_start is not None:
        merged.append((current_start, current_end, gi_id))

    # Map each gene row to its merged GI ID
    def assign_gi(row):
        gs, ge = int(row["start"]), int(row["end"])
        for ms, me, gid in merged:
            if gs <= me and ge >= ms:       # gene overlaps merged interval
                return gid, ms, me
        return 0, 0, 0

    df[["GI_ID", "merged_island_start", "merged_island_end"]] = df.apply(
        assign_gi, axis=1, result_type="expand"
    )
    df["island_start"] = df["merged_island_start"]
    df["island_end"]   = df["merged_island_end"]
    df["island_length"] = df["island_end"] - df["island_start"]
    df = df.drop(columns=["merged_island_start", "merged_island_end"])
    df["GI_flag"] = 1

    # ── Mobility flags from product field ─────────────────────────────────
    flags = df["product"].apply(flag_mobility)
    df["integrase_flag"]   = [f[0] for f in flags]
    df["transposase_flag"] = [f[1] for f in flags]
    df["recombinase_flag"] = [f[2] for f in flags]
    df["mobility_score"]   = (df["integrase_flag"] | df["transposase_flag"] | df["recombinase_flag"]).astype(int)

    print(f"  [IV] {len(df):,} genes across {df['GI_ID'].nunique()} merged GI intervals")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — LOAD AND VALIDATE ABRICATE OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def load_abricate(path: Path, genome_id: str) -> pd.DataFrame:
    """
    Load ABRicate CSV for one genome.
    Applies MIN_IDENTITY and MIN_COVERAGE filters.
    
    Expected columns (standard ABRicate TSV/CSV):
      sample, SEQUENCE, START, END, STRAND, GENE,
      PERCENT_COVERAGE, PERCENT_IDENTITY, PRODUCT
    """
    print(f"  [AB] Loading ABRicate: {path.name}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    # Filter to this genome only (for merged files)
    norm_id = normalise_genome_id(genome_id)
    if "sample" in df.columns:
        df["sample_norm"] = df["sample"].apply(normalise_genome_id)
        df = df[df["sample_norm"] == norm_id].copy()

    if len(df) == 0:
        print(f"  [AB] WARNING: No ABRicate hits found for genome {genome_id}")
        return pd.DataFrame(columns=["contig", "amr_start", "amr_end", "amr_gene"])

    # Quality filters
    before = len(df)
    df = df[
        (df["PERCENT_IDENTITY"] >= MIN_IDENTITY) &
        (df["PERCENT_COVERAGE"] >= MIN_COVERAGE)
    ].copy()
    print(f"  [AB] {len(df):,} / {before:,} AMR hits passed quality filters "
          f"(identity ≥{MIN_IDENTITY}%, coverage ≥{MIN_COVERAGE}%)")

    # Standardise column names
    rename = {
        "SEQUENCE":         "contig",
        "START":            "amr_start",
        "END":              "amr_end",
        "GENE":             "amr_gene",
        "PRODUCT":          "amr_product",
        "PERCENT_IDENTITY": "amr_pid",
        "PERCENT_COVERAGE": "amr_pcov",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["genome_id"] = norm_id
    return df[["genome_id", "contig", "amr_start", "amr_end", "amr_gene",
               "amr_product", "amr_pid", "amr_pcov"]].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — BUILD CASSETTE FEATURE MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(iv_df: pd.DataFrame, ab_df: pd.DataFrame,
                         genome_id: str) -> pd.DataFrame:
    """
    Merge IslandViewer gene table with ABRicate hits to produce
    the cassette-level feature matrix.
    
    Each row = one gene within a GI-anchored cassette.
    """
    if len(iv_df) == 0:
        print("  [FEAT] Cannot build features — no GI genes found.")
        return pd.DataFrame()

    df = iv_df.copy()
    norm_id = normalise_genome_id(genome_id)

    # ── Flag AMR hits by coordinate overlap ──────────────────────────────
    # For each gene, check if any ABRicate hit overlaps it (same contig, ≥1bp)
    df["amr_hit"]     = 0
    df["amr_gene"]    = ""
    df["amr_product"] = ""
    df["amr_pid"]     = 0.0
    df["amr_pcov"]    = 0.0

    if len(ab_df) > 0 and "contig" in ab_df.columns:
        # Use interval overlap: gene [start, end] vs AMR [amr_start, amr_end]
        for _, amr_row in ab_df.iterrows():
            contig = amr_row.get("contig", "")
            a_s    = int(amr_row.get("amr_start", 0))
            a_e    = int(amr_row.get("amr_end", 0))

            # Genes on this contig that overlap with the AMR hit
            # (IslandViewer doesn't carry contig — use coordinate overlap only)
            overlap_mask = (
                (df["start"] <= a_e) & (df["end"] >= a_s)
            )
            if overlap_mask.any():
                df.loc[overlap_mask, "amr_hit"]     = 1
                df.loc[overlap_mask, "amr_gene"]    = str(amr_row.get("amr_gene", ""))
                df.loc[overlap_mask, "amr_product"] = str(amr_row.get("amr_product", ""))
                df.loc[overlap_mask, "amr_pid"]     = float(amr_row.get("amr_pid", 0))
                df.loc[overlap_mask, "amr_pcov"]    = float(amr_row.get("amr_pcov", 0))

    print(f"  [FEAT] AMR-flagged genes: {df['amr_hit'].sum():,} / {len(df):,}")

    # ── Per-GI aggregate features ─────────────────────────────────────────
    gi_stats = df.groupby("GI_ID").agg(
        island_gene_count    = ("locus_tag", "count"),
        amr_gene_count_in_island = ("amr_hit", "sum"),
        Mobility_Load        = ("mobility_score", "sum"),
    ).reset_index()

    gi_stats["GI_AMR_density"] = (
        gi_stats["amr_gene_count_in_island"] / gi_stats["island_gene_count"]
    ).fillna(0)

    df = df.merge(gi_stats, on="GI_ID", how="left")

    # ── AMR neighborhood score (±3-gene window, same logic as training) ───
    df = df.sort_values(["GI_ID", "start"]).reset_index(drop=True)
    df["AMR_neighborhood_score"] = 0.0
    df["AMR_cluster_size"]       = 0

    for gi_id, group in df.groupby("GI_ID"):
        idx_list = list(group.index)
        for pos, idx in enumerate(idx_list):
            if df.at[idx, "amr_hit"] != 1:
                continue
            lo   = max(0, pos - 3)
            hi   = min(len(idx_list), pos + 4)
            win  = df.loc[idx_list[lo:hi]]
            score = (
                2.0 * int(win["integrase_flag"].any()) +
                2.0 * int(win["transposase_flag"].any()) +
                1.0 * int(win["recombinase_flag"].any()) +
                1.0 * int(int(win["amr_hit"].sum()) >= 2)
            )
            df.at[idx, "AMR_neighborhood_score"] = score
            df.at[idx, "AMR_cluster_size"]       = int(win["amr_hit"].sum())

    print(f"  [FEAT] Feature matrix: {len(df):,} rows × {len(df.columns)} columns")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — RUN MODEL AND AGGREGATE TO GENOME LEVEL
# ══════════════════════════════════════════════════════════════════════════════

def predict_genome(df: pd.DataFrame, model, genome_id: str) -> dict:
    """
    Run the trained XGBoost model on cassette feature vectors.
    Returns a dict with genome-level risk score and cassette-level details.
    """
    if len(df) == 0:
        return {"genome_id": genome_id, "error": "No features built — check input files"}

    # ── Select and order model features ───────────────────────────────────
    missing_features = [f for f in MODEL_FEATURES if f not in df.columns]
    if missing_features:
        print(f"  [MODEL] WARNING: Missing features, filling with 0: {missing_features}")
        for f in missing_features:
            df[f] = 0

    X = df[MODEL_FEATURES].fillna(0).values

    # ── Cassette-level probabilities ──────────────────────────────────────
    cassette_probs = model.predict_proba(X)[:, 1]   # probability of class 1 (pathogenic)
    df = df.copy()
    df["cassette_pred_score"] = cassette_probs

    # ── Genome-level aggregation (mean predicted risk across cassettes) ───
    genome_score = float(np.mean(cassette_probs))
    label        = "PATHOGENIC" if genome_score >= 0.5 else "commensal"
    confidence   = genome_score if genome_score >= 0.5 else (1 - genome_score)

    # ── Top high-risk cassettes ───────────────────────────────────────────
    top_cassettes = (
        df[["GI_ID", "locus_tag", "product", "cassette_pred_score",
            "amr_hit", "amr_gene", "integrase_flag", "transposase_flag"]]
        .sort_values("cassette_pred_score", ascending=False)
        .head(10)
    )

    return {
        "genome_id":        normalise_genome_id(genome_id),
        "genome_risk_score": round(genome_score, 4),
        "prediction":       label,
        "confidence":       round(confidence * 100, 1),
        "n_cassette_rows":  len(df),
        "n_gi_intervals":   df["GI_ID"].nunique(),
        "n_amr_genes":      int(df["amr_hit"].sum()),
        "cassette_details": df,
        "top_cassettes":    top_cassettes,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

def save_outputs(result: dict, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    gid = result["genome_id"].replace("/", "_")

    # Summary line
    summary = {k: v for k, v in result.items()
               if k not in ("cassette_details", "top_cassettes")}
    summary_df = pd.DataFrame([summary])
    summary_path = outdir / f"{gid}_prediction_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  [OUT] Summary saved → {summary_path}")

    # Full cassette-level scores
    if "cassette_details" in result and len(result["cassette_details"]) > 0:
        cassette_path = outdir / f"{gid}_cassette_scores.csv"
        result["cassette_details"].to_csv(cassette_path, index=False)
        print(f"  [OUT] Cassette scores saved → {cassette_path}")


# ══════════════════════════════════════════════════════════════════════════════
# PRINT RESULT
# ══════════════════════════════════════════════════════════════════════════════

def print_result(result: dict):
    print()
    print("=" * 58)
    print(f"  CASSETTE2VEC-EC v1.1  —  PREDICTION RESULT")
    print("=" * 58)
    print(f"  Genome ID        : {result['genome_id']}")
    print(f"  Genome Risk Score: {result['genome_risk_score']:.4f}  (0=commensal, 1=pathogenic)")
    print(f"  Prediction       : {result['prediction']}")
    print(f"  Confidence       : {result['confidence']}%")
    print(f"  GI intervals     : {result['n_gi_intervals']}")
    print(f"  Cassette rows    : {result['n_cassette_rows']:,}")
    print(f"  AMR genes in GIs : {result['n_amr_genes']}")
    print()
    if "top_cassettes" in result:
        print("  Top 5 highest-risk cassette loci:")
        top5 = result["top_cassettes"].head(5)
        for _, row in top5.iterrows():
            amr = f" [{row['amr_gene']}]" if row["amr_hit"] else ""
            mob = " [MOB]" if (row["integrase_flag"] or row["transposase_flag"]) else ""
            print(f"    {row['locus_tag']:25s}  score={row['cassette_pred_score']:.3f}"
                  f"  {str(row['product'])[:40]}{amr}{mob}")
    print("=" * 58)
    print()


# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    repo_root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Cassette2Vec-EC v1.1 — Predict pathogenicity of a new E. cecorum genome",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Single genome mode
    p.add_argument("--genome_id",    type=str,  help="Genome accession (e.g. GCF_000379745.1)")
    p.add_argument("--islandviewer", type=Path, help="IslandViewer CSV for this genome")
    p.add_argument("--abricate",     type=Path, help="ABRicate CSV for this genome")

    # Batch mode
    p.add_argument("--batch",              action="store_true", help="Batch mode: process a folder")
    p.add_argument("--islandviewer_dir",   type=Path, help="Folder of IslandViewer CSVs")
    p.add_argument("--abricate_dir",       type=Path, help="Folder of ABRicate CSVs")

    # Shared
    p.add_argument("--model",   type=Path,
                   default=repo_root / "data" / "cassette2vec_v11_model.pkl",
                   help="Path to trained model .pkl (default: data/cassette2vec_v11_model.pkl)")
    p.add_argument("--outdir",  type=Path,
                   default=repo_root / "predictions",
                   help="Output directory (default: predictions/)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Risk score threshold for PATHOGENIC classification (default: 0.5)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_single(genome_id, iv_path, ab_path, model, outdir, threshold):
    print(f"\n[GENOME] Processing: {genome_id}")
    iv_df  = load_islandviewer(iv_path, genome_id)
    ab_df  = load_abricate(ab_path, genome_id)
    feat_df = build_feature_matrix(iv_df, ab_df, genome_id)
    result  = predict_genome(feat_df, model, genome_id)

    # Override threshold if not 0.5
    if threshold != 0.5:
        score = result.get("genome_risk_score", 0)
        result["prediction"]  = "PATHOGENIC" if score >= threshold else "commensal"
        result["confidence"]  = round((score if score >= threshold else 1 - score) * 100, 1)

    print_result(result)
    save_outputs(result, outdir)
    return result


def main():
    args = parse_args()

    # ── Load model ──────────────────────────────────────────────────────────
    if not args.model.exists():
        sys.exit(f"ERROR: Model not found: {args.model}\n"
                 "       Download from Zenodo: https://doi.org/10.5281/zenodo.18529389")
    print(f"[INFO] Loading model: {args.model}")
    model = joblib.load(args.model)

    # ── Batch mode ──────────────────────────────────────────────────────────
    if args.batch:
        if not args.islandviewer_dir or not args.abricate_dir:
            sys.exit("ERROR: --batch requires --islandviewer_dir and --abricate_dir")

        iv_files = sorted(args.islandviewer_dir.glob("*.csv"))
        if not iv_files:
            sys.exit(f"ERROR: No CSV files found in {args.islandviewer_dir}")

        all_results = []
        for iv_path in iv_files:
            genome_id = iv_path.stem.replace("_islands", "")
            ab_path = args.abricate_dir / f"{genome_id}_abricate.csv"
            if not ab_path.exists():
                # Try merged file
                ab_path = args.abricate_dir / "abricate_all_merged.csv"
            if not ab_path.exists():
                print(f"  [SKIP] No ABRicate file for {genome_id}")
                continue
            result = run_single(genome_id, iv_path, ab_path, model,
                                args.outdir, args.threshold)
            all_results.append({k: v for k, v in result.items()
                                 if k not in ("cassette_details", "top_cassettes")})

        # Save batch summary
        if all_results:
            batch_df = pd.DataFrame(all_results).sort_values(
                "genome_risk_score", ascending=False
            )
            batch_path = args.outdir / "batch_prediction_summary.csv"
            batch_df.to_csv(batch_path, index=False)
            print(f"\n[BATCH] Summary saved → {batch_path}")
            print(f"[BATCH] {len(all_results)} genomes processed")
            patho = (batch_df["prediction"] == "PATHOGENIC").sum()
            print(f"[BATCH] {patho} predicted PATHOGENIC / "
                  f"{len(all_results) - patho} commensal")

    # ── Single genome mode ──────────────────────────────────────────────────
    else:
        if not args.genome_id or not args.islandviewer or not args.abricate:
            sys.exit("ERROR: Single mode requires --genome_id, --islandviewer, --abricate\n"
                     "       Or use --batch mode for multiple genomes.")
        run_single(args.genome_id, args.islandviewer, args.abricate,
                   model, args.outdir, args.threshold)


if __name__ == "__main__":
    main()
