"""
predict.py — CASSETTE2VEC v1
Simple prediction script to load a trained model and predict
pathogenic vs commensal classification for a given feature vector.

This is a placeholder for v1. Real FASTA → feature extraction
will be added in future versions.
"""

import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path


def load_model(model_path="saved_models/xgboost_model.pkl"):
    """Load a trained CASSETTE2VEC model."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def predict_from_csv(model, feature_csv):
    """
    Predict pathogenicity from a pre-computed feature table (1 row).

    Parameters
    ----------
    model : sklearn/xgboost model
    feature_csv : str

    Returns
    -------
    dict
        {"probability": float, "prediction": "pathogenic"/"commensal"}
    """
    df = pd.read_csv(feature_csv)

    # ensure no genome_id column
    if "genome_id" in df.columns:
        df = df.drop(columns=["genome_id"])

    prob = model.predict_proba(df)[0][1]
    pred = "pathogenic" if prob >= 0.5 else "commensal"

    return {"probability": float(prob), "prediction": pred}


def main():
    if len(sys.argv) != 3:
        print("Usage: python predict.py <model.pkl> <feature_table.csv>")
        sys.exit(1)

    model_path = sys.argv[1]
    csv_path = sys.argv[2]

    model = load_model(model_path)
    result = predict_from_csv(model, csv_path)

    print("\n=== CASSETTE2VEC Prediction ===")
    print(f"Prediction: {result['prediction']}")
    print(f"Probability (pathogenic): {result['probability']:.4f}")


if __name__ == "__main__":
    main()
