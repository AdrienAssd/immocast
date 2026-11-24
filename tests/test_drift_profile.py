import json

import numpy as np
import pandas as pd
import pytest

from ml import policy
from ml.drift import bin_shares, build_profile, compare_distributions, drift_report


def apartments(rows=2000, seed=0, central=False):
    rng = np.random.default_rng(seed)
    arrondissements = [6, 7, 8] if central else list(range(1, 21))
    return pd.DataFrame({
        "surface": rng.uniform(100, 250, rows) if central else rng.gamma(4, 14, rows).clip(9, 400),
        "pieces": rng.integers(4, 8, rows) if central else rng.integers(1, 6, rows),
        "arrondissement": rng.choice(arrondissements, rows),
    })


def test_profile_has_shares_that_sum_to_one_for_every_feature():
    profile = build_profile(apartments())

    assert set(profile["features"]) == {"surface", "pieces", "arrondissement"}
    for spec in profile["features"].values():
        assert sum(spec["shares"]) == pytest.approx(1.0)
    json.dumps(profile)


@pytest.mark.parametrize(
    "reference, current, expected",
    [
        pytest.param([0.5, 0.5], [0.5, 0.5], 0.0, id="identical"),
        pytest.param([1.0, 0.0], [0.0, 1.0], 1.0, id="no-overlap"),
        pytest.param([0.6, 0.4], [0.4, 0.6], 0.2, id="partial-shift"),
    ],
)
def test_compare_distributions_is_the_total_variation_distance(reference, current, expected):
    assert compare_distributions(reference, current) == pytest.approx(expected)


def test_same_population_has_low_drift():
    profile = build_profile(apartments(seed=0))
    report = drift_report(profile, apartments(seed=1))

    assert max(report.values()) < policy.DRIFT_THRESHOLD


def test_large_central_apartments_are_flagged_as_drift():
    profile = build_profile(apartments())
    report = drift_report(profile, apartments(seed=1, central=True))

    assert all(value >= policy.DRIFT_THRESHOLD for value in report.values())


def test_values_outside_the_training_range_fall_in_the_edge_bins():
    spec = build_profile(apartments())["features"]["surface"]
    shares = bin_shares([1, 10_000], spec)

    assert shares[0] == pytest.approx(0.5)
    assert shares[-1] == pytest.approx(0.5)


def test_no_values_gives_no_shares():
    spec = build_profile(apartments())["features"]["pieces"]
    assert sum(bin_shares([], spec)) == 0


def test_drift_threshold_matches_the_course_policy():
    assert policy.DRIFT_THRESHOLD == 0.20
