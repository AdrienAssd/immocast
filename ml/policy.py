from dataclasses import dataclass

METRIC = "mae"
HIGHER_IS_BETTER = False

QUALITY_FLOOR_MAE = 2300.0
RETRAIN_DEGRADATION = 0.05
MIN_IMPROVEMENT = 0.03
DRIFT_THRESHOLD = 0.20


@dataclass(frozen=True)
class Decision:
    passed: bool
    reason: str


def meets_quality_floor(mae: float) -> Decision:
    if mae <= QUALITY_FLOOR_MAE:
        return Decision(True, f"MAE {mae:.0f} is within the quality floor of {QUALITY_FLOOR_MAE:.0f}")
    return Decision(False, f"MAE {mae:.0f} is above the quality floor of {QUALITY_FLOOR_MAE:.0f}")


def should_retrain(champion_mae: float, approved_mae: float) -> Decision:
    if champion_mae > QUALITY_FLOOR_MAE:
        return Decision(True, f"champion MAE {champion_mae:.0f} is above the quality floor of {QUALITY_FLOOR_MAE:.0f}")
    degradation = (champion_mae - approved_mae) / approved_mae
    if degradation >= RETRAIN_DEGRADATION:
        return Decision(
            True,
            f"champion MAE degraded by {degradation:.1%} since approval "
            f"({approved_mae:.0f} -> {champion_mae:.0f}), threshold {RETRAIN_DEGRADATION:.0%}",
        )
    return Decision(
        False,
        f"champion MAE changed by {degradation:+.1%} since approval "
        f"({approved_mae:.0f} -> {champion_mae:.0f}), below the {RETRAIN_DEGRADATION:.0%} retraining threshold",
    )


def should_promote(candidate_mae: float, champion_mae: float) -> Decision:
    floor = meets_quality_floor(candidate_mae)
    if not floor.passed:
        return Decision(False, f"candidate rejected: {floor.reason}")
    improvement = (champion_mae - candidate_mae) / champion_mae
    if improvement < MIN_IMPROVEMENT:
        return Decision(
            False,
            f"candidate rejected: improvement {improvement:+.1%} ({champion_mae:.0f} -> {candidate_mae:.0f}) "
            f"is below the required {MIN_IMPROVEMENT:.0%}",
        )
    return Decision(
        True,
        f"candidate promoted: improvement {improvement:+.1%} ({champion_mae:.0f} -> {candidate_mae:.0f}), "
        f"required {MIN_IMPROVEMENT:.0%}",
    )
