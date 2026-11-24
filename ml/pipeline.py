import argparse
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import pandas as pd
import policy
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from model import FEATURES, TARGET, build_model, evaluate, model_params, train
from train import dvc_hash, git_commit

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_NAME = os.environ.get("MODEL_NAME", "immocast")
EXPERIMENT = os.environ.get("MLFLOW_CT_EXPERIMENT_NAME", "immocast-ct")
OUTPUT_DIR = ROOT / os.environ.get("RUN_OUTPUT_DIR", "results")
CANDIDATE_MODEL = "hgb"
ALIAS = "champion"

REQUIRED_COLUMNS = ["id_mutation", "date_mutation", *FEATURES, TARGET]
RANGES = {
    "surface": (9, 400),
    "pieces": (1, 10),
    "arrondissement": (1, 20),
    TARGET: (3000, 30000),
}
MIN_ROWS = 1000


class ValidationError(Exception):
    pass


def load_split(version: str, split: str) -> pd.DataFrame:
    path = DATA_DIR / version / f"{split}.csv"
    if not path.exists():
        raise ValidationError(f"{path.relative_to(ROOT)} is missing (run dvc pull)")
    return pd.read_csv(path)


def validate_split(df: pd.DataFrame, name: str) -> list[str]:
    errors = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return [f"{name}: missing columns {missing}"]
    if len(df) < MIN_ROWS:
        errors.append(f"{name}: {len(df)} rows, at least {MIN_ROWS} required")
    nulls = df[REQUIRED_COLUMNS].isna().sum()
    for column, count in nulls[nulls > 0].items():
        errors.append(f"{name}: {count} missing values in {column}")
    if df["id_mutation"].duplicated().any():
        errors.append(f"{name}: duplicated id_mutation")
    for column, (low, high) in RANGES.items():
        values = pd.to_numeric(df[column], errors="coerce")
        bad = (~values.between(low, high)).sum()
        if bad:
            errors.append(f"{name}: {bad} values of {column} outside [{low}, {high}]")
    if not (df["pieces"] % 1 == 0).all():
        errors.append(f"{name}: pieces must be whole numbers")
    if pd.to_datetime(df["date_mutation"], errors="coerce").isna().any():
        errors.append(f"{name}: invalid date_mutation")
    return errors


def validate(version: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train_df = load_split(version, "train")
    eval_df = load_split(version, "evaluation")
    errors = validate_split(train_df, "train") + validate_split(eval_df, "evaluation")
    if not errors:
        overlap = set(train_df["id_mutation"]) & set(eval_df["id_mutation"])
        if overlap:
            errors.append(f"{len(overlap)} id_mutation present in both train and evaluation")
        train_end = pd.to_datetime(train_df["date_mutation"]).max()
        eval_start = pd.to_datetime(eval_df["date_mutation"]).min()
        if train_end >= eval_start:
            errors.append(f"evaluation starts on {eval_start.date()} before the end of training {train_end.date()}")
    if errors:
        raise ValidationError("; ".join(errors))
    summary = {
        "train_rows": len(train_df),
        "evaluation_rows": len(eval_df),
        "train_period": [train_df["date_mutation"].min(), train_df["date_mutation"].max()],
        "evaluation_period": [eval_df["date_mutation"].min(), eval_df["date_mutation"].max()],
    }
    return train_df, eval_df, summary


def champion_version(client: MlflowClient):
    try:
        return client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    except MlflowException:
        return None


def log_candidate(model, train_df: pd.DataFrame, register: bool):
    sample = train_df[FEATURES].head(5)
    info = mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        signature=infer_signature(sample, model.predict(sample)),
        input_example=sample,
        registered_model_name=MODEL_NAME if register else None,
    )
    return info


def registered_version(client: MlflowClient, run_id: str) -> str:
    versions = [v for v in client.search_model_versions(f"name='{MODEL_NAME}'") if v.run_id == run_id]
    return str(max(versions, key=lambda v: int(v.version)).version)


def approve(client: MlflowClient, version: str, mae: float, dataset_version: str):
    client.set_model_version_tag(MODEL_NAME, version, "approved", "true")
    client.set_model_version_tag(MODEL_NAME, version, "approved_mae", f"{mae:.4f}")
    client.set_model_version_tag(MODEL_NAME, version, "dataset_version", dataset_version)
    client.set_model_version_tag(MODEL_NAME, version, "dvc_hash", dvc_hash(dataset_version))
    client.set_model_version_tag(MODEL_NAME, version, "approved_at", datetime.now(UTC).isoformat(timespec="seconds"))
    client.set_registered_model_alias(MODEL_NAME, ALIAS, version)


def write_outputs(result: dict, artifacts: dict[str, pd.DataFrame] | None = None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "decision.json").write_text(json.dumps(result, indent=2, default=str))
    manifests = OUTPUT_DIR / "manifests"
    if result.get("dataset_version"):
        manifests.mkdir(exist_ok=True)
        pointer = DATA_DIR / f"{result['dataset_version']}.dvc"
        if pointer.exists():
            shutil.copy(pointer, manifests / pointer.name)
    for name, df in (artifacts or {}).items():
        df.to_csv(OUTPUT_DIR / name, index=False)
    if mlflow.active_run():
        mlflow.log_artifacts(str(OUTPUT_DIR), artifact_path="decision")
    print(json.dumps(result, indent=2, default=str))


def run_pipeline(dataset_version: str) -> int:
    client = MlflowClient()
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"ct-{dataset_version}") as run:
        result = {
            "operation": "pipeline",
            "run_id": run.info.run_id,
            "model_name": MODEL_NAME,
            "dataset_version": dataset_version,
            "dvc_hash": dvc_hash(dataset_version),
            "git_commit": git_commit(),
            "metric": policy.METRIC,
            "higher_is_better": policy.HIGHER_IS_BETTER,
            "gates": {
                "quality_floor_mae": policy.QUALITY_FLOOR_MAE,
                "retrain_degradation": policy.RETRAIN_DEGRADATION,
                "min_improvement": policy.MIN_IMPROVEMENT,
            },
        }
        mlflow.set_tags({k: result[k] for k in ("operation", "dataset_version", "dvc_hash", "git_commit", "model_name")})
        mlflow.log_params({f"gate_{k}": v for k, v in result["gates"].items()})

        try:
            train_df, eval_df, summary = validate(dataset_version)
        except ValidationError as e:
            result.update(validation="failed", decision="validation_failed", reason=str(e), retrained=False)
            mlflow.set_tags({"validation": "failed", "decision": "validation_failed"})
            write_outputs(result)
            return 1
        result.update(validation="passed", data=summary)
        mlflow.set_tag("validation", "passed")
        mlflow.log_params({"train_rows": summary["train_rows"], "evaluation_rows": summary["evaluation_rows"]})

        champion = champion_version(client)
        result["champion_before"] = str(champion.version) if champion else None
        artifacts = {}

        if champion:
            champion_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{champion.version}")
            champion_metrics, champion_details = evaluate(champion_model, eval_df)
            approved_mae = float(champion.tags.get("approved_mae", champion_metrics["mae"]))
            result.update(champion_mae=champion_metrics["mae"], champion_approved_mae=approved_mae)
            mlflow.log_metrics({"champion_mae": champion_metrics["mae"], "champion_approved_mae": approved_mae})
            artifacts["champion_predictions.csv"] = champion_details

            retrain = policy.should_retrain(champion_metrics["mae"], approved_mae)
            result["retrain_reason"] = retrain.reason
            if not retrain.passed:
                result.update(retrained=False, decision="retained", reason=retrain.reason,
                              champion_after=result["champion_before"])
                mlflow.set_tags({"decision": "retained", "champion_after": result["champion_before"]})
                write_outputs(result, artifacts)
                return 0
        else:
            result["retrain_reason"] = "no champion yet: bootstrap the first model"

        mlflow.log_params({"candidate_model": CANDIDATE_MODEL, **model_params(CANDIDATE_MODEL)})
        candidate = train(build_model(CANDIDATE_MODEL), train_df)
        candidate_metrics, candidate_details = evaluate(candidate, eval_df)
        result.update(retrained=True, candidate_mae=candidate_metrics["mae"], candidate_mape=candidate_metrics["mape"])
        mlflow.log_metrics({"candidate_mae": candidate_metrics["mae"], "candidate_mape": candidate_metrics["mape"]})
        artifacts["candidate_predictions.csv"] = candidate_details

        if champion:
            gate = policy.should_promote(candidate_metrics["mae"], result["champion_mae"])
        else:
            gate = policy.meets_quality_floor(candidate_metrics["mae"])

        log_candidate(candidate, train_df, register=gate.passed)
        if gate.passed:
            version = registered_version(client, run.info.run_id)
            approve(client, version, candidate_metrics["mae"], dataset_version)
            decision = "promoted" if champion else "bootstrapped"
            result.update(decision=decision, reason=gate.reason, candidate_version=version, champion_after=version)
        else:
            result.update(decision="candidate_rejected", reason=gate.reason,
                          champion_after=result["champion_before"])
        mlflow.set_tags({"decision": result["decision"], "champion_after": str(result["champion_after"])})
        write_outputs(result, artifacts)
        return 0 if (gate.passed or champion) else 1


def run_rollback(target: str) -> int:
    client = MlflowClient()
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"rollback-{target}") as run:
        champion = champion_version(client)
        result = {
            "operation": "rollback",
            "run_id": run.info.run_id,
            "model_name": MODEL_NAME,
            "git_commit": git_commit(),
            "champion_before": str(champion.version) if champion else None,
            "requested_version": target,
        }
        mlflow.set_tags({"operation": "rollback", "requested_version": target})
        try:
            version = client.get_model_version(MODEL_NAME, target)
        except MlflowException:
            version = None

        if version is None:
            reason = f"version {target} of {MODEL_NAME} does not exist"
        elif version.tags.get("approved") != "true":
            reason = f"version {target} of {MODEL_NAME} was never approved"
        elif result["champion_before"] == target:
            reason = f"version {target} is already the champion"
        else:
            reason = None

        if reason:
            result.update(decision="rollback_rejected", reason=reason, champion_after=result["champion_before"])
            mlflow.set_tags({"decision": "rollback_rejected"})
            write_outputs(result)
            return 1

        client.set_registered_model_alias(MODEL_NAME, ALIAS, target)
        result.update(decision="rolled_back", champion_after=target,
                      reason=f"champion moved from {result['champion_before']} to {target}",
                      dataset_version=version.tags.get("dataset_version"))
        mlflow.set_tags({"decision": "rolled_back", "champion_after": target})
        write_outputs(result)
        return 0


def main():
    parser = argparse.ArgumentParser(description="Continuous training: validate, evaluate, retrain, promote or roll back.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset-version")
    group.add_argument("--rollback", metavar="MODEL_VERSION")
    args = parser.parse_args()

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    if args.rollback:
        sys.exit(run_rollback(args.rollback))
    sys.exit(run_pipeline(args.dataset_version))


if __name__ == "__main__":
    main()
