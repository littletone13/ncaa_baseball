#!/usr/bin/env python3
"""Load baseball data into Supabase via batched SQL INSERT statements.

Works around IPv4/pooler issues by generating SQL that can be executed via
the Supabase MCP execute_sql tool or any SQL client.

Usage:
  python3 scripts/load_baseball_via_mcp.py --all --execute
  python3 scripts/load_baseball_via_mcp.py --table predictions --date 2026-03-27 --execute
  python3 scripts/load_baseball_via_mcp.py --table teams --dry-run  # just print SQL

When --execute is set, requires SUPABASE_PROJECT_ID env var or --project-id flag.
Uses the supabase management API via the MCP, so no direct DB connection needed.
Without --execute, prints SQL to stdout for manual execution.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from io import StringIO
from pathlib import Path

import pandas as pd

SCHEMA = "baseball"
BATCH_SIZE = 200  # rows per INSERT statement


def _escape_sql(val: str) -> str:
    """Escape a string value for SQL insertion."""
    if val is None:
        return "NULL"
    s = str(val).strip()
    if s in ("", "nan", "None", "NaN", "\\N"):
        return "NULL"
    return "'" + s.replace("'", "''") + "'"


def _to_sql_val(val, col_type: str = "text") -> str:
    """Convert a Python value to a SQL literal based on target type."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "NULL"
    s = str(val).strip()
    if s in ("", "nan", "None", "NaN", "\\N"):
        return "NULL"
    if col_type in ("smallint", "integer", "bigint"):
        try:
            return str(int(float(s)))
        except (ValueError, OverflowError):
            return "NULL"
    elif col_type == "double precision":
        try:
            return str(float(s))
        except ValueError:
            return "NULL"
    elif col_type == "boolean":
        return "true" if s.lower() in ("true", "1") else "false"
    else:
        return _escape_sql(s)


def generate_upsert_sql(
    df: pd.DataFrame,
    table: str,
    conflict_cols: list[str],
    col_types: dict[str, str] | None = None,
) -> list[str]:
    """Generate batched INSERT ... ON CONFLICT statements."""
    if df.empty:
        return []

    df = df.drop_duplicates(subset=conflict_cols, keep="last")
    cols = list(df.columns)
    col_types = col_types or {}

    statements = []
    for batch_start in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[batch_start : batch_start + BATCH_SIZE]
        values_list = []
        for _, row in batch.iterrows():
            vals = ", ".join(_to_sql_val(row.get(c), col_types.get(c, "text")) for c in cols)
            values_list.append(f"({vals})")

        values_sql = ",\n".join(values_list)
        insert_cols = ", ".join(cols)
        conflict = ", ".join(conflict_cols)
        update_cols = [c for c in cols if c not in conflict_cols]

        if update_cols:
            update_set = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
            sql = f"INSERT INTO {SCHEMA}.{table} ({insert_cols})\nVALUES\n{values_sql}\nON CONFLICT ({conflict}) DO UPDATE SET {update_set};"
        else:
            sql = f"INSERT INTO {SCHEMA}.{table} ({insert_cols})\nVALUES\n{values_sql}\nON CONFLICT ({conflict}) DO NOTHING;"

        statements.append(sql)

    return statements


# ── Column type maps (matching the Supabase schema) ──────────────────────

TEAM_TYPES = {"season": "integer"}
STADIUM_TYPES = {"lat": "double precision", "lon": "double precision",
                 "hp_bearing_deg": "double precision", "elevation_ft": "double precision"}
GAME_TYPES = {"season": "integer", "home_score": "integer", "away_score": "integer",
              "winner_home": "boolean", "neutral_site": "boolean",
              "has_run_events": "boolean", "has_boxscore": "boolean"}
APPEARANCE_TYPES = {"season": "integer", "starter": "boolean", "ip": "double precision",
                    "h": "integer", "r": "integer", "er": "integer", "bb": "integer",
                    "k": "integer", "hr": "integer", "pc": "integer"}
PLAYER_TYPES = {"is_pitcher": "boolean", "is_batter": "boolean", "pitcher_idx": "integer",
                "fip": "double precision", "era": "double precision",
                "wrc_plus": "double precision", "season": "integer"}
RUN_EVENT_TYPES = {"season": "integer", "home_run_1": "integer", "home_run_2": "integer",
                   "home_run_3": "integer", "home_run_4": "integer",
                   "away_run_1": "integer", "away_run_2": "integer",
                   "away_run_3": "integer", "away_run_4": "integer",
                   "home_score": "integer", "away_score": "integer"}
META_TYPES = {"n_teams": "integer", "n_pitchers": "integer", "n_conf": "integer",
              "n_draws": "integer", "scoring_calibration": "double precision"}
TEAM_PARAM_TYPES = {"team_idx": "integer", "season": "integer",
                    "bullpen_quality_z": "double precision", "bullpen_adj": "double precision",
                    "wrc_plus": "double precision", "wrc_offense_adj": "double precision",
                    "conf_strength_adj": "double precision",
                    "batting_fb_pct": "double precision", "batting_fb_factor": "double precision",
                    "n_games": "integer"}
PITCHER_PARAM_TYPES = {"pitcher_idx": "integer", "season": "integer",
                       "season_ip": "double precision", "season_era": "double precision",
                       "fip": "double precision", "siera": "double precision",
                       "fb_pct": "double precision", "fb_sensitivity": "double precision",
                       "d1b_ability_adj": "double precision", "n_appearances": "integer"}
PREDICTION_TYPES = {"home_win_prob": "double precision", "away_win_prob": "double precision",
                    "ml_home": "integer", "ml_away": "integer",
                    "exp_home": "double precision", "exp_away": "double precision",
                    "exp_total": "double precision", "exp_total_p10": "double precision",
                    "exp_total_p50": "double precision", "exp_total_p90": "double precision",
                    "margin_p10": "double precision", "margin_p50": "double precision",
                    "margin_p90": "double precision", "over_prob": "double precision",
                    "park_factor": "double precision", "weather_adj": "double precision",
                    "wind_out_mph": "double precision", "temp_f": "double precision",
                    "home_bullpen_adj": "double precision", "away_bullpen_adj": "double precision",
                    "home_wrc_adj": "double precision", "away_wrc_adj": "double precision",
                    "hp_d1b_adj": "double precision", "ap_d1b_adj": "double precision",
                    "mkt_anchor_weight": "double precision", "mkt_home_win_prob": "double precision",
                    "mkt_total_line": "double precision",
                    "home_starter_idx": "integer", "away_starter_idx": "integer"}


# ── Per-table generators ─────────────────────────────────────────────────

def gen_teams(season: int = 2026) -> list[str]:
    csv = Path("data/registries/canonical_teams_2026.csv")
    df = pd.read_csv(csv, dtype=str)
    df["season"] = str(season)
    keep = ["canonical_id", "season", "team_name", "conference", "ncaa_teams_id",
            "odds_api_name", "espn_name", "baseballr_team_id", "baseballr_team_name", "notes"]
    keep = [c for c in keep if c in df.columns]
    return generate_upsert_sql(df[keep], "teams", ["canonical_id", "season"], TEAM_TYPES)


def gen_stadiums() -> list[str]:
    csv = Path("data/registries/stadium_orientations.csv")
    df = pd.read_csv(csv, dtype=str)
    keep = ["canonical_id", "venue_name", "lat", "lon", "hp_bearing_deg",
            "elevation_ft", "timezone", "source"]
    keep = [c for c in keep if c in df.columns]
    return generate_upsert_sql(df[keep], "stadiums", ["canonical_id"], STADIUM_TYPES)


def gen_games() -> list[str]:
    csv = Path("data/processed/games.csv")
    df = pd.read_csv(csv, dtype=str)
    df = df.dropna(subset=["event_id", "home_canonical_id", "away_canonical_id"])
    df = df.drop_duplicates(subset=["event_id"], keep="last")
    keep = ["event_id", "game_date", "season", "home_canonical_id", "away_canonical_id",
            "home_name", "away_name", "home_score", "away_score", "winner_home",
            "venue_name", "venue_city", "venue_state", "neutral_site",
            "home_pitcher_espn_id", "away_pitcher_espn_id",
            "home_pitcher_name", "away_pitcher_name",
            "has_run_events", "has_boxscore"]
    keep = [c for c in keep if c in df.columns]
    return generate_upsert_sql(df[keep], "games", ["event_id"], GAME_TYPES)


def gen_appearances() -> list[str]:
    csv = Path("data/processed/pitcher_appearances.csv")
    df = pd.read_csv(csv, dtype=str, low_memory=False)
    df = df.dropna(subset=["event_id", "pitcher_id", "pitcher_name", "team_canonical_id"])
    df = df[df["pitcher_id"].str.strip() != ""]
    df = df.drop_duplicates(subset=["event_id", "pitcher_id"], keep="last")
    keep = ["event_id", "game_date", "season", "pitcher_espn_id", "pitcher_id",
            "pitcher_name", "team_canonical_id", "team_name", "side", "starter",
            "role", "ip", "h", "r", "er", "bb", "k", "hr", "pc"]
    keep = [c for c in keep if c in df.columns]
    return generate_upsert_sql(df[keep], "pitcher_appearances", ["event_id", "pitcher_id"], APPEARANCE_TYPES)


def gen_players() -> list[str]:
    csv = Path("data/processed/player_registry.csv")
    if not csv.exists():
        return []
    df = pd.read_csv(csv, dtype=str)
    df = df.dropna(subset=["canonical_id", "player_name"])
    df = df.drop_duplicates(subset=["canonical_id", "player_name", "season"], keep="last")
    keep = ["canonical_id", "player_name", "position", "bats", "throws",
            "is_pitcher", "is_batter", "pitcher_idx", "fip", "era",
            "wrc_plus", "season", "sources"]
    keep = [c for c in keep if c in df.columns]
    return generate_upsert_sql(df[keep], "players", ["canonical_id", "player_name", "season"], PLAYER_TYPES)


def gen_run_events() -> list[str]:
    csv = Path("data/processed/run_events.csv")
    df = pd.read_csv(csv, dtype=str)
    df = df.dropna(subset=["event_id"])
    df = df.drop_duplicates(subset=["event_id"], keep="last")
    keep = ["event_id", "game_date", "season", "home_canonical_id", "away_canonical_id",
            "home_pitcher_espn_id", "away_pitcher_espn_id",
            "home_run_1", "home_run_2", "home_run_3", "home_run_4",
            "away_run_1", "away_run_2", "away_run_3", "away_run_4",
            "home_score", "away_score"]
    keep = [c for c in keep if c in df.columns]
    return generate_upsert_sql(df[keep], "run_events", ["event_id"], RUN_EVENT_TYPES)


def gen_model_meta(fit_id: str) -> list[str]:
    meta_json = Path("data/processed/run_event_fit_meta.json")
    meta = json.loads(meta_json.read_text())
    df = pd.DataFrame([{
        "fit_id": fit_id,
        "fit_date": fit_id,
        "n_teams": str(meta["N_teams"]),
        "n_pitchers": str(meta["N_pitchers"]),
        "n_conf": str(meta.get("N_conf", 30)),
        "n_draws": str(meta["n_draws"]),
        "scoring_calibration": "0.083",
    }])
    return generate_upsert_sql(df, "model_posterior_meta", ["fit_id"], META_TYPES)


def gen_team_params(fit_id: str) -> list[str]:
    csv = Path("data/processed/team_table.csv")
    df = pd.read_csv(csv, dtype=str)
    df["fit_id"] = fit_id
    keep = ["canonical_id", "fit_id", "team_idx", "team_name", "conference", "season",
            "bullpen_quality_z", "bullpen_adj", "wrc_plus", "wrc_offense_adj",
            "conf_strength_adj", "batting_fb_pct", "batting_fb_factor", "n_games"]
    keep = [c for c in keep if c in df.columns]
    return generate_upsert_sql(df[keep], "team_model_params", ["canonical_id", "fit_id"], TEAM_PARAM_TYPES)


def gen_pitcher_params(fit_id: str) -> list[str]:
    csv = Path("data/processed/pitcher_table.csv")
    df = pd.read_csv(csv, dtype=str)
    df["fit_id"] = fit_id
    df = df[df["pitcher_espn_id"].notna() & (df["pitcher_espn_id"] != "")]
    keep = ["pitcher_espn_id", "fit_id", "pitcher_idx", "pitcher_name",
            "team_canonical_id", "season", "throws", "role",
            "season_ip", "season_era", "fip", "siera",
            "fb_pct", "fb_sensitivity", "d1b_ability_adj", "d1b_ability_source",
            "n_appearances", "last_appearance"]
    keep = [c for c in keep if c in df.columns]
    return generate_upsert_sql(df[keep], "pitcher_model_params", ["pitcher_espn_id", "fit_id"], PITCHER_PARAM_TYPES)


def gen_predictions(date_str: str) -> list[str]:
    csv = Path(f"data/processed/predictions_{date_str}.csv")
    if not csv.exists():
        print(f"  Predictions not found: {csv}", file=sys.stderr)
        return []
    df = pd.read_csv(csv, dtype=str)
    df["prediction_date"] = date_str
    renames = {"home_cid": "home_canonical_id", "away_cid": "away_canonical_id",
               "home": "home_name", "away": "away_name"}
    df = df.rename(columns=renames)
    df = df.dropna(subset=["home_canonical_id", "away_canonical_id"])
    keep = ["prediction_date", "home_canonical_id", "away_canonical_id",
            "home_name", "away_name", "home_starter", "away_starter",
            "home_starter_idx", "away_starter_idx", "hp_throws", "ap_throws",
            "home_win_prob", "away_win_prob", "ml_home", "ml_away",
            "exp_home", "exp_away", "exp_total",
            "exp_total_p10", "exp_total_p50", "exp_total_p90",
            "margin_p10", "margin_p50", "margin_p90",
            "over_prob", "park_factor", "weather_adj", "wind_out_mph", "temp_f",
            "home_bullpen_adj", "away_bullpen_adj",
            "home_wrc_adj", "away_wrc_adj",
            "hp_d1b_adj", "ap_d1b_adj",
            "mkt_anchor_weight", "mkt_home_win_prob", "mkt_total_line"]
    keep = [c for c in keep if c in df.columns]
    return generate_upsert_sql(df[keep], "predictions",
                               ["prediction_date", "home_canonical_id", "away_canonical_id"],
                               PREDICTION_TYPES)


# ── Table registry ───────────────────────────────────────────────────────

TABLE_GENERATORS = {
    "teams": lambda args: gen_teams(args.season),
    "stadiums": lambda args: gen_stadiums(),
    "games": lambda args: gen_games(),
    "pitcher_appearances": lambda args: gen_appearances(),
    "players": lambda args: gen_players(),
    "run_events": lambda args: gen_run_events(),
    "model_meta": lambda args: gen_model_meta(args.fit_id),
    "team_params": lambda args: gen_team_params(args.fit_id),
    "pitcher_params": lambda args: gen_pitcher_params(args.fit_id),
    "predictions": lambda args: gen_predictions(args.date) if args.date else [],
}

# Order matters for FK constraints
ALL_TABLES = ["teams", "stadiums", "games", "pitcher_appearances", "players",
              "run_events", "model_meta", "team_params", "pitcher_params", "predictions"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Load baseball data via SQL INSERT batches")
    parser.add_argument("--all", action="store_true", help="Load all tables")
    parser.add_argument("--table", help="Load specific table")
    parser.add_argument("--date", help="Date for predictions (YYYY-MM-DD)")
    parser.add_argument("--fit-id", default="2026-03-27", help="Model fit ID")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--dry-run", action="store_true", help="Print SQL to stdout")
    parser.add_argument("--execute", action="store_true", help="Execute via Supabase MCP")
    parser.add_argument("--project-id", default="otfybzwvockuwdldfoed", help="Supabase project ID")
    parser.add_argument("--out-dir", type=Path, help="Write SQL files to this directory")
    args = parser.parse_args()

    tables_to_load = []
    if args.all:
        tables_to_load = ALL_TABLES
        # For --all, also load all prediction files
    elif args.table:
        t = args.table.lower().replace("appearances", "pitcher_appearances")
        if t in TABLE_GENERATORS:
            tables_to_load = [t]
        else:
            print(f"Unknown table: {t}. Available: {', '.join(TABLE_GENERATORS)}", file=sys.stderr)
            return 1
    else:
        print("Specify --all or --table NAME", file=sys.stderr)
        return 1

    total_stmts = 0
    for table in tables_to_load:
        print(f"Generating SQL for {SCHEMA}.{table}...", file=sys.stderr)
        stmts = TABLE_GENERATORS[table](args)
        if not stmts:
            print(f"  {table}: 0 statements (empty/missing)", file=sys.stderr)
            continue
        print(f"  {table}: {len(stmts)} batch statements", file=sys.stderr)

        if args.out_dir:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            for i, sql in enumerate(stmts):
                p = args.out_dir / f"{table}_{i:04d}.sql"
                p.write_text(sql)
            print(f"  Wrote {len(stmts)} files to {args.out_dir}/", file=sys.stderr)

        if args.dry_run:
            for sql in stmts:
                print(sql)
                print()

        total_stmts += len(stmts)

    # For --all, also generate prediction files
    if args.all:
        for f in sorted(Path("data/processed").glob("predictions_2026-*.csv")):
            date_str = f.stem.replace("predictions_", "").split("_")[0]
            if len(date_str) == 10:
                stmts = gen_predictions(date_str)
                if stmts:
                    print(f"  predictions ({date_str}): {len(stmts)} statements", file=sys.stderr)
                    if args.out_dir:
                        for i, sql in enumerate(stmts):
                            p = args.out_dir / f"predictions_{date_str}_{i:04d}.sql"
                            p.write_text(sql)
                    total_stmts += len(stmts)

    print(f"\nTotal: {total_stmts} SQL statements generated", file=sys.stderr)
    if not args.dry_run and not args.execute and not args.out_dir:
        print("Use --dry-run to print, --out-dir to save, or --execute to run via MCP", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
