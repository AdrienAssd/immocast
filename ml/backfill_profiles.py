"""Adds a training profile to approved model versions that were approved before profiles existed.

python ml/backfill_profiles.py            add missing profiles (needs dvc pull of the versions)
python ml/backfill_profiles.py --dry-run  only list what would be done
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from drift import build_profile
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = os.environ.get("MODEL_NAME", "immocast")
PROFILE_FILE = "training-profile.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    client = MlflowClient()
    for version in sorted(client.search_model_versions(f"name='{MODEL_NAME}'"), key=lambda v: int(v.version)):
        tags = version.tags
        if tags.get("approved") != "true":
            continue
        if tags.get("approval_run_id"):
            print(f"version {version.version}: profile already recorded")
            continue
        dataset_version = tags.get("dataset_version")
        train_path = ROOT / "data" / str(dataset_version) / "train.csv"
        if not train_path.exists():
            raise SystemExit(f"version {version.version}: {train_path} is missing, run dvc pull first")
        print(f"version {version.version}: profile from {dataset_version}/train.csv added to run {version.run_id}")
        if args.dry_run:
            continue
        profile = build_profile(pd.read_csv(train_path))
        client.log_dict(version.run_id, profile, PROFILE_FILE)
        client.set_model_version_tag(MODEL_NAME, version.version, "approval_run_id", version.run_id)


if __name__ == "__main__":
    main()
