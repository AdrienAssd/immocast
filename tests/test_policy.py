import math

import pytest

from ml import policy


def test_gates_match_readme_experiments():
    assert policy.METRIC == "mae"
    assert policy.HIGHER_IS_BETTER is False
    assert policy.QUALITY_FLOOR_MAE == 2300
    assert policy.RETRAIN_DEGRADATION == 0.05
    assert policy.MIN_IMPROVEMENT == 0.03


@pytest.mark.parametrize(
    "mae, expected",
    [
        pytest.param(1898.0, True, id="v1-candidate-bootstraps"),
        pytest.param(2300.0, True, id="exactly-at-the-floor"),
        pytest.param(2300.5, False, id="just-above-the-floor"),
        pytest.param(3100.0, False, id="far-above-the-floor"),
        pytest.param(math.nan, False, id="nan-mae"),
    ],
)
def test_quality_floor(mae, expected):
    decision = policy.meets_quality_floor(mae)

    assert decision.passed is expected
    assert decision.reason


@pytest.mark.parametrize(
    "champion_mae, approved_mae, expected",
    [
        pytest.param(2102.0, 1898.0, True, id="v2-champion-degraded-10.7pct"),
        pytest.param(2100.0, 2000.0, True, id="degradation-exactly-5pct"),
        pytest.param(2098.0, 2000.0, False, id="degradation-4.9pct"),
        pytest.param(2000.0, 2000.0, False, id="unchanged"),
        pytest.param(1985.0, 2019.0, False, id="v3-champion-improved"),
        pytest.param(2350.0, 2290.0, True, id="above-floor-even-with-small-degradation"),
    ],
)
def test_retraining_rule(champion_mae, approved_mae, expected):
    decision = policy.should_retrain(champion_mae, approved_mae)

    assert decision.passed is expected
    assert decision.reason


@pytest.mark.parametrize(
    "candidate_mae, champion_mae, expected",
    [
        pytest.param(2019.0, 2102.0, True, id="v2-candidate-improves-3.9pct"),
        pytest.param(1940.0, 2000.0, True, id="improvement-exactly-3pct"),
        pytest.param(1942.0, 2000.0, False, id="improvement-2.9pct"),
        pytest.param(2000.0, 2000.0, False, id="same-mae"),
        pytest.param(2100.0, 2000.0, False, id="worse-than-champion"),
        pytest.param(2350.0, 2700.0, False, id="big-improvement-but-above-floor"),
        pytest.param(math.nan, 2000.0, False, id="nan-candidate"),
    ],
)
def test_promotion_rule(candidate_mae, champion_mae, expected):
    decision = policy.should_promote(candidate_mae, champion_mae)

    assert decision.passed is expected
    assert ("candidate promoted" if expected else "candidate rejected") in decision.reason


def test_documented_continuous_training_scenario():
    assert policy.meets_quality_floor(1898.0).passed
    assert policy.should_retrain(2102.0, 1898.0).passed
    assert policy.should_promote(2019.0, 2102.0).passed
    assert not policy.should_retrain(1985.0, 2019.0).passed
