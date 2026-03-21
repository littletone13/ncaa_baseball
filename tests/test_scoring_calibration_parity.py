from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from ncaa_baseball.model_runtime import (
    SCORING_CALIBRATION,
    assert_scoring_calibration_parity,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_globals(rel_path: str) -> dict:
    return runpy.run_path(str(REPO_ROOT / rel_path), run_name="__test__")


def test_simulate_uses_shared_scoring_calibration() -> None:
    g = _load_script_globals("scripts/simulate.py")
    assert g["SIMULATE_SCORING_CALIBRATION"] == SCORING_CALIBRATION


def test_backtest_uses_shared_scoring_calibration() -> None:
    g = _load_script_globals("scripts/backtest_posterior.py")
    assert g["BACKTEST_SCORING_CALIBRATION"] == SCORING_CALIBRATION


def test_scoring_calibration_parity_assertion_raises_on_divergence() -> None:
    with pytest.raises(RuntimeError, match="SCORING_CALIBRATION divergence"):
        assert_scoring_calibration_parity("unit-test", SCORING_CALIBRATION + 0.001)
