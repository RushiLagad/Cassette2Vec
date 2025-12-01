"""
train.py — baseline training script for CASSETTE2VEC v1

Usage (locally, not on GitHub):

    python -m cassette2vec.models.train \
        --input data/master/cassette2vec_master.csv \
        --target_col label \
        --model_out models/c2v_xgb_model.pkl
"""

import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train baseline XGBoost model for CASSETTE2VEC v1."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to master feature CSV (one row per genome)."
    )
    parser.add_argument(
        "--target_col", default="label",
        help="Name of the target column (e.g. 'label' with 0=commensal, 1=pathogenic)."
    )
    parser.add_argument(
        "--model_out", default="models/c2v_xgb_model.pkl",
        help="Where to save the trained model."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load data
    df = pd.read_csv(args.input)

    if args.target_col not in df.columns:
        raise ValueError(
            f"Target column '{args.target_col}' not found in input file. "
            "Please make sure your master CSV has this column."
        )

    y = df[args.target_col].astype(int)
    X = df.drop(columns=[args.target_col])

    # Simple train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Baseline XGBoost model (you can tune later)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # Evaluation
    probas = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, probas)
    pr_auc = average_precision_score(y_test, probas)

    print(f"[CASSETTE2VEC] ROC-AUC: {roc:.3f}")
    print(f"[CASSETTE2VEC] PR-AUC : {pr_auc:.3f}")
    print(f"[CASSETTE2VEC] Test size: {len(y_test)} genomes")

    # Save model
    joblib.dump(model, args.model_out)
    print(f"[CASSETTE2VEC] Saved model to: {args.model_out}")


if __name__ == "__main__":
    main()
