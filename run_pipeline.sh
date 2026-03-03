#!/bin/bash
set -e
echo "[1/3] Adding AMR neighborhood features..."
python add_AMR_neighborhood_v1.py

echo "[2/3] Generating evaluation figures + SHAP..."
python cassette2vec_ML_v11_FINAL.py

echo "[3/3] Done. Figures saved to figures/"
