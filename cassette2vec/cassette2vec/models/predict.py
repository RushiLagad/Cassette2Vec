
"""
CASSETTE2VEC v1 - Prediction script

Usage (example):
    python cassette2vec/models/predict.py \
        --features data/new_genomes_features.csv \
        --model cassette2vec_xgb.pkl \
        --output predictions.csv
"""

import argparse
import joblib
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Predict pathogenicity with CASSETTE2VEC v1.")
    parser.add_argument(
        "--features",
        required=True,
        help="CSV file with feature columns (same as training, without label).",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model .pkl file (from train.py).",
    )
    parser.add_argument(
        "--output",
        default="predictions.csv",
        help="Output CSV file with predictions.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[INFO] Loading model from: {args.model}")
    artifact = joblib.load(args.model)
    model = artifact["model"]
    feature_names = artifact["feature_names"]

    print(f"[INFO] Reading features from: {args.features}")
    df = pd.read_csv(args.features)

    # Ensure all required features are present
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required feature columns in input CSV: {missing}"
        )

    X = df[feature_names]

    print(f"[INFO] Running predictions on {len(X)} genomes...")
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    out = df.copy()
    out["cassette2vec_score"] = proba
    out["cassette2vec_pred"] = pred

    print(f"[INFO] Writing predictions to: {args.output}")
    out.to_csv(args.output, index=False)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
