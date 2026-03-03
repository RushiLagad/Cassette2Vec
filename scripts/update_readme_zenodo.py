#!/usr/bin/env python3
"""
update_readme_zenodo.py
=======================
Queries the Zenodo REST API for record 18529389 and automatically
updates the Data section of README.md with live file information.

Called by GitHub Actions (.github/workflows/update_zenodo.yml).
Can also be run locally:
    python scripts/update_readme_zenodo.py
"""

import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ZENODO_RECORD_ID = "18529389"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
README_PATH = Path(__file__).resolve().parent.parent / "README.md"

FILE_DESCRIPTIONS = {
    "cassette2vec_v11_model.pkl": "Trained XGBoost model — load with `joblib.load()`",
    "cassette2vec_ML_features_v1_with_mobility_load.csv": "Base cassette feature matrix (145 genomes, 51,302 rows)",
    "cassette2vec_v11_predictions.csv": "Cassette-level model predictions (`true_label`, `pred_score`)",
    "cassette2vec_islandviewer_all_clean.csv": "IslandViewer genomic island calls (all 145 genomes, cleaned)",
    "cassette2vec_predict.py": "Prediction pipeline — run on a new genome",
    "cassette2vec_ML_v11_FINAL.py": "Evaluation + SHAP figure generation script",
    "add_AMR_neighborhood_v1.py": "Step 2 pipeline — adds AMR neighborhood scores to feature matrix",
    "requirements.txt": "Pinned Python package versions",
    "environment.yml": "Conda environment specification",
}


def human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.2f} {unit}" if unit != "B" else f"{n_bytes} B"
        n_bytes /= 1024
    return f"{n_bytes:.2f} GB"


def fetch_zenodo_record() -> dict:
    print(f"[INFO] Fetching Zenodo record {ZENODO_RECORD_ID}...")
    req = urllib.request.Request(
        ZENODO_API_URL,
        headers={"User-Agent": "Cassette2Vec-README-Updater/1.0 (github.com/RushiLagad/Cassette2Vec)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        sys.exit(f"ERROR: Zenodo API returned {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        sys.exit(f"ERROR: Could not reach Zenodo API: {e.reason}")
    return data


def build_files_table(record: dict) -> str:
    files = record.get("files", [])
    if not files:
        return "_No files found in this Zenodo record._\n"

    def sort_key(f):
        name = f.get("key", "")
        if name.endswith(".py"): return (0, name)
        if name.endswith(".txt"): return (1, name)
        if name.endswith(".yml"): return (2, name)
        if name.endswith(".zip"): return (3, name)
        return (4, -f.get("size", 0))

    files_sorted = sorted(files, key=sort_key)
    lines = [
        "| File | Size | Description | Download |",
        "|------|------|-------------|----------|",
    ]
    for f in files_sorted:
        name = f.get("key", "unknown")
        size = human_size(f.get("size", 0))
        desc = FILE_DESCRIPTIONS.get(name, "—")
        link = f.get("links", {}).get("self", "")
        dl_badge = f"[⬇ Download]({link})" if link else "—"
        lines.append(f"| `{name}` | {size} | {desc} | {dl_badge} |")
    return "\n".join(lines) + "\n"


def build_zenodo_section(record: dict) -> str:
    doi = record.get("doi", "10.5281/zenodo.18529389")
    version = record.get("metadata", {}).get("version", "v1.0.0")
    stats = record.get("stats", {})
    views = stats.get("unique_views", stats.get("views", "—"))
    downloads = stats.get("unique_downloads", stats.get("downloads", "—"))
    pub_date = record.get("metadata", {}).get("publication_date", "—")
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_files = len(record.get("files", []))
    total_bytes = sum(f.get("size", 0) for f in record.get("files", []))
    total_size = human_size(total_bytes)
    table = build_files_table(record)

    section = f"""
<!-- ZENODO_FILES_START -->
> **Auto-generated from Zenodo API** — Last updated: {updated}

### 📦 Zenodo Archive

| | |
|---|---|
| **DOI** | [![DOI](https://zenodo.org/badge/DOI/{doi}.svg)](https://doi.org/{doi}) |
| **Version** | {version} |
| **Published** | {pub_date} |
| **Files** | {n_files} files ({total_size} total) |
| **Views / Downloads** | {views} / {downloads} |

### 📥 Download Files

{table}
### 🚀 Quick Download (command line)

```bash
pip install zenodo-get
zenodo_get {ZENODO_RECORD_ID} -o data/
```
<!-- ZENODO_FILES_END -->
"""
    return section


def update_readme(section: str) -> bool:
    if not README_PATH.exists():
        sys.exit(f"ERROR: README not found at {README_PATH}")
    content = README_PATH.read_text(encoding="utf-8")
    start_marker = "<!-- ZENODO_FILES_START -->"
    end_marker = "<!-- ZENODO_FILES_END -->"
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    if start_idx == -1 or end_idx == -1:
        print("[WARN] Markers not found in README. Appending section at end.")
        new_content = content.rstrip() + "\n\n" + section + "\n"
    else:
        end_idx += len(end_marker)
        new_content = content[:start_idx] + section.strip() + "\n" + content[end_idx:]
    if new_content == content:
        print("[INFO] README already up to date — no changes made.")
        return False
    README_PATH.write_text(new_content, encoding="utf-8")
    print(f"[SAVED] README updated: {README_PATH}")
    return True


def main():
    record = fetch_zenodo_record()
    section = build_zenodo_section(record)
    changed = update_readme(section)
    files = record.get("files", [])
    print(f"\n[INFO] Record: {record.get('doi', '—')}")
    print(f"[INFO] Version: {record.get('metadata', {}).get('version', '—')}")
    print(f"[INFO] Files found: {len(files)}")
    for f in files:
        print(f"       {f.get('key', '?'):60s} {human_size(f.get('size', 0))}")
    print(f"\n[DONE] README {'updated' if changed else 'already current'}.")


if __name__ == "__main__":
    main()
