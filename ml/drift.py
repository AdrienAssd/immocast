"""Training profile and drift measure shared by the pipeline and the API.

A profile stores, for each feature, the bins and the share of training rows in each bin.
Drift is the total variation distance between the training shares and the shares observed
on new rows: 0 means identical distributions, 1 means no overlap.
"""

import numpy as np
import pandas as pd

PROFILE_VERSION = 1
DISCRETE_FEATURES = {"pieces": list(range(1, 11)), "arrondissement": list(range(1, 21))}
NUMERIC_FEATURES = ["surface"]
NUMERIC_BINS = 10


def _numeric_edges(values: pd.Series) -> list[float]:
    quantiles = np.quantile(values, np.linspace(0, 1, NUMERIC_BINS + 1))
    return sorted({float(q) for q in quantiles[1:-1]})


def bin_shares(values, spec: dict) -> list[float]:
    values = pd.Series(values, dtype=float).dropna()
    if values.empty:
        return [0.0] * (len(spec["values"]) if spec["kind"] == "discrete" else len(spec["edges"]) + 1)
    if spec["kind"] == "discrete":
        counts = values.round().astype(int).value_counts()
        shares = [float(counts.get(v, 0)) for v in spec["values"]]
    else:
        positions = np.searchsorted(spec["edges"], values.to_numpy(), side="right")
        shares = np.bincount(positions, minlength=len(spec["edges"]) + 1).astype(float).tolist()
    total = sum(shares)
    return [s / total for s in shares] if total else shares


def build_profile(df: pd.DataFrame) -> dict:
    features = {}
    for name, values in DISCRETE_FEATURES.items():
        spec = {"kind": "discrete", "values": values}
        features[name] = {**spec, "shares": bin_shares(df[name], spec)}
    for name in NUMERIC_FEATURES:
        spec = {"kind": "numeric", "edges": _numeric_edges(df[name])}
        features[name] = {**spec, "shares": bin_shares(df[name], spec)}
    return {"profile_version": PROFILE_VERSION, "rows": len(df), "features": features}


def compare_distributions(reference: list[float], current: list[float]) -> float:
    return round(0.5 * float(np.abs(np.asarray(reference) - np.asarray(current)).sum()), 4)


def drift_report(profile: dict, df: pd.DataFrame) -> dict[str, float]:
    return {
        name: compare_distributions(spec["shares"], bin_shares(df[name], spec))
        for name, spec in profile["features"].items()
        if name in df.columns
    }
