"""
CASSETTE2VEC – prediction script

This is a minimal, generic interface:
- loads a trained scikit-learn model from disk
- loads a feature table for one or more genomes
- prints pathogenicity probabilities

Note: The actual trained model file is NOT shipped in the repo.
Users should train their own model using train.py.
"""

import argparse
import pathlib
import sys
from typing import List

import joblib
import pandas as pd


def load_model(model_path: pathlib.Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Train a model first using train.py and save it as a .pkl file."
        )
    return joblib.load(model_path)


def load_features(feature_path: pathlib.Path, id_columns: List[str]):
    df = pd.read_csv(feature_path)
    meta_cols = [c for c in id_columns if c in df.columns]
    X = df.drop(columns=[c for c in id_columns if c in df.columns], errors="ignore")
    return df, X, meta_cols


def main():
    parser = argparse.ArgumentParser(
        description="Predict pathogenic vs commensal genomes using CASSETTE2VEC."
    )
    parser.add_argument(
        "--features",
        required=True,
        help="Path to CSV file with model-ready features (one row per genome).",
    )
    parser.add_argument(
        "--model",
        default="cassette2vec_xgb.pkl",
        help="Path to trained model .pkl file (default: cassette2vec_xgb.pkl in current directory).",
    )
    parser.add_argument(
        "--id-columns",
        nargs="*",
        default=["genome_id"],
        help="Columns to treat as identifiers (not used as features).",
    )
    args = parser.parse_args()

    feature_path = pathlib.Path(args.features)
    model_path = pathlib.Path(args.model)

    try:
        model = load_model(model_path)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    df, X, meta_cols = load_features(feature_path, args.id_columns)

    # Get prediction probabilities (assumes binary classifier with predict_proba)
    try:
        proba = model.predict_proba(X)[:, 1]
    except AttributeError:
        # Fallback if model has no predict_proba
        proba = model.predict(X)

    # Build output table
    out = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)
    out["cassette2vec_pathogenic_prob"] = proba

    # Print to stdout (can be redirected to a file)
    out.to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    main()
