import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from model import FEATURES, MODELS, build_model, evaluate, model_params, train

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT_NAME", "immocast")


def git_commit() -> str:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def dvc_hash(version: str) -> str:
    pointer = DATA_DIR / f"{version}.dvc"
    if not pointer.exists():
        return "unknown"
    match = re.search(r"md5:\s*(\S+)", pointer.read_text())
    return match.group(1) if match else "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-version", default="v1")
    parser.add_argument("--model", choices=list(MODELS), default="rf")
    parser.add_argument("--register", metavar="MODEL_NAME")
    parser.add_argument("--set-champion", action="store_true")
    args = parser.parse_args()

    train_df = pd.read_csv(DATA_DIR / args.dataset_version / "train.csv")
    eval_df = pd.read_csv(DATA_DIR / args.dataset_version / "evaluation.csv")

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"{args.model}-{args.dataset_version}") as run:
        mlflow.set_tags({
            "git_commit": git_commit(),
            "dataset_version": args.dataset_version,
            "dvc_hash": dvc_hash(args.dataset_version),
            "model_type": args.model,
        })
        mlflow.log_params({
            "model": args.model,
            "dataset_version": args.dataset_version,
            "features": ",".join(FEATURES),
            "train_rows": len(train_df),
            "train_start": train_df["date_mutation"].min(),
            "train_end": train_df["date_mutation"].max(),
            "evaluation_start": eval_df["date_mutation"].min(),
            "evaluation_end": eval_df["date_mutation"].max(),
            **model_params(args.model),
        })

        model = train(build_model(args.model), train_df)
        metrics, details = evaluate(model, eval_df)
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmp:
            details.to_csv(Path(tmp) / "evaluation_predictions.csv", index=False)
            (Path(tmp) / "metrics.json").write_text(json.dumps(metrics, indent=2))
            mlflow.log_artifacts(tmp, artifact_path="evaluation")

        sample = train_df[FEATURES].head(5)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=infer_signature(sample, model.predict(sample)),
            input_example=sample,
            registered_model_name=args.register,
        )

        result = {"run_id": run.info.run_id, "model": args.model, "dataset_version": args.dataset_version, **metrics}

        if args.register:
            client = MlflowClient()
            version = max(
                (v for v in client.search_model_versions(f"name='{args.register}'") if v.run_id == run.info.run_id),
                key=lambda v: int(v.version),
            ).version
            result["registered_model"] = args.register
            result["registered_version"] = version
            if args.set_champion:
                client.set_registered_model_alias(args.register, "champion", version)
                result["champion"] = version
            mlflow.set_tag("registered_version", version)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "model.joblib")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
