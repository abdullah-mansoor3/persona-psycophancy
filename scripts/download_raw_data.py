#!/usr/bin/env python3
"""Download the raw Twin-2K-500 data files (wave_split config) into data/raw/.

Files produced (all gitignored, shared via download):
  data/raw/wave_split/wave_persona_chunk_001..007.parquet   (HF: wave_split/chunks/...)
  data/raw/wave1_3_response_label.csv
  data/raw/wave4_response_label.csv
  data/raw/question_catalog.json

Requires: pip install huggingface_hub
Run from the repository root:  python scripts/download_raw_data.py

Dataset: LLM-Digital-Twin/Twin-2K-500 (configuration `wave_split`).
The 7 parquet chunks together hold all 2,058 participants and the central
`wave_split` columns used by the project (persona text/JSON, wave4 question
fields, pid). The two *_response_label.csv files are the official flattened
human answers, and question_catalog.json maps every QID to its question
metadata and flattened CSV columns. Sizes of downloaded files are verified
against the HF listing; add `--force` to re-download.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "LLM-Digital-Twin/Twin-2K-500"
REVISION = "main"

# (repo path, local destination relative to repo root)
FILES = [
    ("wave_split/chunks/wave_persona_chunk_001.parquet", "data/raw/wave_split/"),
    ("wave_split/chunks/wave_persona_chunk_002.parquet", "data/raw/wave_split/"),
    ("wave_split/chunks/wave_persona_chunk_003.parquet", "data/raw/wave_split/"),
    ("wave_split/chunks/wave_persona_chunk_004.parquet", "data/raw/wave_split/"),
    ("wave_split/chunks/wave_persona_chunk_005.parquet", "data/raw/wave_split/"),
    ("wave_split/chunks/wave_persona_chunk_006.parquet", "data/raw/wave_split/"),
    ("wave_split/chunks/wave_persona_chunk_007.parquet", "data/raw/wave_split/"),
    ("question_catalog_and_human_response_csv/wave1_3_response_label.csv", "data/raw/"),
    ("question_catalog_and_human_response_csv/wave4_response_label.csv", "data/raw/"),
    ("question_catalog_and_human_response_csv/question_catalog.json", "data/raw/"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if files exist")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    missing = []
    for repo_path, dest_rel in FILES:
        dest = repo_root / dest_rel / Path(repo_path).name
        if dest.exists() and not args.force:
            continue
        missing.append((repo_path, dest))

    if not missing:
        print("All raw files already present under data/raw/. Use --force to re-download.")
        return 0

    print(f"Downloading {len(missing)} file(s) from {REPO_ID} (revision {REVISION}) ...")
    for repo_path, dest in missing:
        print(f"  -> {repo_path}")
        local = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=repo_path, revision=REVISION)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, dest)

    print("Done. Verify expected sizes with:")
    print("  ls -la data/raw data/raw/wave_split")
    print("Then build the processed dataset:")
    print("  python src/data_pipeline/build_processed_dataset.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
