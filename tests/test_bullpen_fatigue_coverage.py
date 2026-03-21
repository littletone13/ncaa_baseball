from __future__ import annotations

from pathlib import Path

import pandas as pd

from bullpen_fatigue import compute_bullpen_fatigue


def _write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_compute_bullpen_fatigue_emits_required_team_rows_when_no_window_data(tmp_path: Path) -> None:
    appearances = tmp_path / "pitcher_appearances.csv"
    _write_csv(
        appearances,
        [
            {
                "game_date": "2026-03-01",
                "team_canonical_id": "TEAM_A",
                "role": "reliever",
                "ip": "1.0",
            }
        ],
    )

    out = compute_bullpen_fatigue(
        appearances_csv=appearances,
        game_date="2026-03-21",
        window_days=3,
        required_team_ids={"TEAM_A", "TEAM_B"},
    )
    assert set(out["canonical_id"]) == {"TEAM_A", "TEAM_B"}
    assert (out["fatigue_adj"] == 0.0).all()
    assert (out["fatigue_flag"] == 0).all()
    assert "fatigue_data_status" in out.columns


def test_compute_bullpen_fatigue_fills_missing_required_teams(tmp_path: Path) -> None:
    appearances = tmp_path / "pitcher_appearances.csv"
    _write_csv(
        appearances,
        [
            {
                "game_date": "2026-03-20",
                "team_canonical_id": "TEAM_A",
                "role": "reliever",
                "ip": "2.1",
            },
            {
                "game_date": "2026-03-20",
                "team_canonical_id": "TEAM_A",
                "role": "starter",
                "ip": "4.0",
            },
        ],
    )

    out = compute_bullpen_fatigue(
        appearances_csv=appearances,
        game_date="2026-03-21",
        window_days=3,
        required_team_ids={"TEAM_A", "TEAM_B"},
    )
    assert set(out["canonical_id"]) == {"TEAM_A", "TEAM_B"}
    imputed = out[out["canonical_id"] == "TEAM_B"].iloc[0]
    assert float(imputed["fatigue_adj"]) == 0.0
    assert int(imputed["fatigue_flag"]) == 0
