from __future__ import annotations

import pytest

from ncaa_baseball.model_runtime import (
    FATIGUE_POLICY_ABORT,
    FATIGUE_POLICY_DERISK,
    FATIGUE_POLICY_IGNORE,
    enforce_fatigue_coverage_policy,
)


def test_fatigue_policy_apply_when_coverage_meets_threshold() -> None:
    decision = enforce_fatigue_coverage_policy(
        required_team_ids={"A", "B", "C"},
        fatigue_team_ids={"A", "B", "C"},
        policy=FATIGUE_POLICY_ABORT,
        min_coverage=0.8,
        context_label="test",
    )
    assert decision.action == "apply"
    assert decision.coverage == 1.0


def test_fatigue_policy_derisk_on_low_coverage() -> None:
    decision = enforce_fatigue_coverage_policy(
        required_team_ids={"A", "B", "C", "D"},
        fatigue_team_ids={"A"},
        policy=FATIGUE_POLICY_DERISK,
        min_coverage=0.75,
        context_label="test",
    )
    assert decision.action == "de-risk"
    assert decision.coverage == 0.25
    assert "de-risking" in decision.message


def test_fatigue_policy_abort_on_low_coverage() -> None:
    with pytest.raises(RuntimeError, match="aborting due to low fatigue coverage"):
        enforce_fatigue_coverage_policy(
            required_team_ids={"A", "B"},
            fatigue_team_ids={"A"},
            policy=FATIGUE_POLICY_ABORT,
            min_coverage=1.0,
            context_label="test",
        )


def test_fatigue_policy_ignore_allows_partial_coverage() -> None:
    decision = enforce_fatigue_coverage_policy(
        required_team_ids={"A", "B", "C"},
        fatigue_team_ids={"A"},
        policy=FATIGUE_POLICY_IGNORE,
        min_coverage=0.9,
        context_label="test",
    )
    assert decision.action == "ignore"
    assert decision.coverage == pytest.approx(1.0 / 3.0)
