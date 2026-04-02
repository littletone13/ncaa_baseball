#!/usr/bin/env python3
"""
load_baseball_to_postgres.py — Bulk load CSV data into Supabase baseball schema.

Usage:
  python3 scripts/load_baseball_to_postgres.py --all
  python3 scripts/load_baseball_to_postgres.py --table teams
  python3 scripts/load_baseball_to_postgres.py --table predictions --date 2026-03-26
  python3 scripts/load_baseball_to_postgres.py --table model_meta --fit-id 2026-03-21
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import psycopg

DEFAULT_DSN = (
    "postgresql://postgres:{}@db.{}.supabase.co:5432/postgres?sslmode=require"
)
PROJECT_REF = "otfybzwvockuwdldfoed"


def get_dsn(args) -> str:
    if args.dsn:
        return args.dsn
    env_dsn = os.environ.get("DATABASE_URL")
    if env_dsn:
        return env_dsn
    pw = os.environ.get("SUPABASE_DB_PASSWORD", "")
    if not pw:
        # Try .env
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("SUPABASE_DB_PASSWORD="):
                    pw = line.split("=", 1)[1].strip().strip("'\"")
    if pw:
        return DEFAULT_DSN.format(pw, PROJECT_REF)
    print("No database connection. Set DATABASE_URL, SUPABASE_DB_PASSWORD, or --dsn.", file=sys.stderr)
    sys.exit(1)


def upsert_from_df(cur, df: pd.DataFrame, table: str, conflict_cols: list[str], schema: str = "baseball"):
    """Bulk upsert a DataFrame via COPY to temp table then INSERT ON CONFLICT."""
    if df.empty:
        print(f"  {schema}.{table}: 0 rows (empty)", file=sys.stderr)
        return 0

    # Always dedupe on conflict columns to avoid "cannot affect row a second time"
    df = df.drop_duplicates(subset=conflict_cols, keep="last")

    cols = list(df.columns)
    tmp = f"_tmp_{table}"

    # Create temp table matching target
    cur.execute(f"DROP TABLE IF EXISTS {tmp}")
    col_defs = ", ".join(f'"{c}" text' for c in cols)
    cur.execute(f"CREATE TEMP TABLE {tmp} ({col_defs})")

    # COPY data in — use CSV format to handle embedded quotes/tabs
    buf = StringIO()
    df.to_csv(buf, index=False, header=False, sep=",", na_rep="\\N", quoting=1)
    buf.seek(0)
    with cur.copy(f"COPY {tmp} ({','.join(cols)}) FROM STDIN WITH (FORMAT csv, NULL '\\N')") as copy:
        while data := buf.read(65536):
            copy.write(data.encode())

    # Build upsert
    conflict = ", ".join(conflict_cols)
    insert_cols = ", ".join(cols)
    select_cols = ", ".join(f'"{c}"' for c in cols)

    # For numeric columns, cast from text
    target_types = _get_column_types(cur, schema, table)
    cast_cols = []
    for c in cols:
        target_type = target_types.get(c, "text")
        if target_type in ("smallint", "integer", "bigint"):
            cast_cols.append(f'CASE WHEN "{c}" = \'\\N\' OR "{c}" = \'\' THEN NULL ELSE "{c}"::double precision::integer END AS "{c}"')
        elif target_type == "double precision":
            cast_cols.append(f'CASE WHEN "{c}" = \'\\N\' OR "{c}" = \'\' THEN NULL ELSE "{c}"::double precision END AS "{c}"')
        elif target_type == "boolean":
            cast_cols.append(f'CASE WHEN "{c}" IN (\'True\', \'true\', \'1\') THEN true WHEN "{c}" IN (\'False\', \'false\', \'0\') THEN false ELSE NULL END AS "{c}"')
        elif target_type == "date":
            cast_cols.append(f'CASE WHEN "{c}" = \'\\N\' OR "{c}" = \'\' THEN NULL ELSE "{c}"::date END AS "{c}"')
        else:
            cast_cols.append(f'CASE WHEN "{c}" = \'\\N\' THEN NULL ELSE "{c}" END AS "{c}"')
    cast_select = ", ".join(cast_cols)

    # Exclude conflict columns from update
    update_cols = [c for c in cols if c not in conflict_cols]
    if update_cols:
        update_set = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in update_cols)
        upsert_sql = f"""
            INSERT INTO {schema}.{table} ({insert_cols})
            SELECT {cast_select} FROM {tmp}
            ON CONFLICT ({conflict}) DO UPDATE SET {update_set}
        """
    else:
        upsert_sql = f"""
            INSERT INTO {schema}.{table} ({insert_cols})
            SELECT {cast_select} FROM {tmp}
            ON CONFLICT ({conflict}) DO NOTHING
        """

    cur.execute(upsert_sql)
    n = cur.rowcount
    cur.execute(f"DROP TABLE IF EXISTS {tmp}")
    print(f"  {schema}.{table}: {n} rows upserted", file=sys.stderr)
    return n


def _get_column_types(cur, schema: str, table: str) -> dict[str, str]:
    cur.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
    """, (schema, table))
    return {row[0]: row[1] for row in cur.fetchall()}


# ── Per-table loaders ──────────────────────────────────────────────────────

def load_teams(cur, season: int = 2026):
    csv = Path("data/registries/canonical_teams_2026.csv")
    df = pd.read_csv(csv, dtype=str)
    df["season"] = str(season)
    keep = ["canonical_id", "season", "team_name", "conference", "ncaa_teams_id",
            "odds_api_name", "espn_name", "baseballr_team_id", "baseballr_team_name", "notes"]
    keep = [c for c in keep if c in df.columns]
    return upsert_from_df(cur, df[keep], "teams", ["canonical_id", "season"])


def load_stadiums(cur):
    csv = Path("data/registries/stadium_orientations.csv")
    df = pd.read_csv(csv, dtype=str)
    keep = ["canonical_id", "venue_name", "lat", "lon", "hp_bearing_deg",
            "elevation_ft", "timezone", "source"]
    keep = [c for c in keep if c in df.columns]
    return upsert_from_df(cur, df[keep], "stadiums", ["canonical_id"])


def load_games(cur):
    csv = Path("data/processed/games.csv")
    df = pd.read_csv(csv, dtype=str)
    # Drop rows missing required fields, dedupe on PK
    df = df.dropna(subset=["event_id", "home_canonical_id", "away_canonical_id"])
    df = df.drop_duplicates(subset=["event_id"], keep="last")
    keep = ["event_id", "game_date", "season", "home_canonical_id", "away_canonical_id",
            "home_name", "away_name", "home_score", "away_score", "winner_home",
            "venue_name", "venue_city", "venue_state", "neutral_site",
            "home_pitcher_espn_id", "away_pitcher_espn_id",
            "home_pitcher_name", "away_pitcher_name",
            "has_run_events", "has_boxscore"]
    keep = [c for c in keep if c in df.columns]
    return upsert_from_df(cur, df[keep], "games", ["event_id"])


def load_pitcher_appearances(cur):
    csv = Path("data/processed/pitcher_appearances.csv")
    df = pd.read_csv(csv, dtype=str)
    # Drop rows missing required fields
    df = df.dropna(subset=["event_id", "pitcher_id", "pitcher_name", "team_canonical_id"])
    df = df[df["pitcher_id"].str.strip() != ""]
    df = df.drop_duplicates(subset=["event_id", "pitcher_id"], keep="last")
    keep = ["event_id", "game_date", "season", "pitcher_espn_id", "pitcher_id",
            "pitcher_name", "team_canonical_id", "team_name", "side", "starter",
            "role", "ip", "h", "r", "er", "bb", "k", "hr", "pc"]
    keep = [c for c in keep if c in df.columns]
    return upsert_from_df(cur, df[keep], "pitcher_appearances", ["event_id", "pitcher_id"])


def load_players(cur):
    csv = Path("data/processed/player_registry.csv")
    df = pd.read_csv(csv, dtype=str)
    df = df.dropna(subset=["canonical_id", "player_name"])
    df = df.drop_duplicates(subset=["canonical_id", "player_name", "season"], keep="last")
    keep = ["canonical_id", "player_name", "position", "bats", "throws",
            "is_pitcher", "is_batter", "pitcher_idx", "fip", "era",
            "wrc_plus", "season", "sources"]
    keep = [c for c in keep if c in df.columns]
    return upsert_from_df(cur, df[keep], "players", ["canonical_id", "player_name", "season"])


def load_run_events(cur):
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
    return upsert_from_df(cur, df[keep], "run_events", ["event_id"])


def load_model_meta(cur, fit_id: str):
    meta_json = Path("data/processed/run_event_fit_meta.json")
    meta = json.loads(meta_json.read_text())
    df = pd.DataFrame([{
        "fit_id": fit_id,
        "fit_date": fit_id,
        "n_teams": str(meta["N_teams"]),
        "n_pitchers": str(meta["N_pitchers"]),
        "n_conf": str(meta.get("N_conf", 30)),
        "n_draws": str(meta["n_draws"]),
        "scoring_calibration": "0.12",
    }])
    return upsert_from_df(cur, df, "model_posterior_meta", ["fit_id"])


def load_team_params(cur, fit_id: str):
    csv = Path("data/processed/team_table.csv")
    df = pd.read_csv(csv, dtype=str)
    df["fit_id"] = fit_id
    keep = ["canonical_id", "fit_id", "team_idx", "team_name", "conference", "season",
            "bullpen_quality_z", "bullpen_adj", "wrc_plus", "wrc_offense_adj",
            "conf_strength_adj", "batting_fb_pct", "batting_fb_factor", "n_games"]
    keep = [c for c in keep if c in df.columns]
    return upsert_from_df(cur, df[keep], "team_model_params", ["canonical_id", "fit_id"])


def load_pitcher_params(cur, fit_id: str):
    csv = Path("data/processed/pitcher_table.csv")
    df = pd.read_csv(csv, dtype=str)
    df["fit_id"] = fit_id
    # Drop rows with no pitcher_espn_id
    df = df[df["pitcher_espn_id"].notna() & (df["pitcher_espn_id"] != "")]
    keep = ["pitcher_espn_id", "fit_id", "pitcher_idx", "pitcher_name",
            "team_canonical_id", "season", "throws", "role",
            "season_ip", "season_era", "fip", "siera",
            "fb_pct", "fb_sensitivity", "d1b_ability_adj", "d1b_ability_source",
            "n_appearances", "last_appearance"]
    keep = [c for c in keep if c in df.columns]
    return upsert_from_df(cur, df[keep], "pitcher_model_params", ["pitcher_espn_id", "fit_id"])


def upload_projections_to_syndicate(date_str: str, predictions_csv: Path | None = None) -> int:
    """
    Upload predictions directly to the syndicate-terminal Supabase projections table.
    Replaces the ncaa_baseball_adapter.py middleman.

    Uses the same SUPABASE_DB_PASSWORD / DATABASE_URL credentials as the rest of this script.
    Upserts on (sport, game_id, model_version='current') so re-runs update in place.
    """
    import re

    if predictions_csv is None:
        predictions_csv = Path(f"data/processed/predictions_{date_str}_standard.csv")
        if not predictions_csv.exists():
            # fallback to non-phase filename
            predictions_csv = Path(f"data/processed/predictions_{date_str}.csv")
    if not predictions_csv.exists():
        print(f"  Projections upload skipped — not found: {predictions_csv}", file=sys.stderr)
        return 0

    def _slugify(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(s).lower().strip()).strip("_")

    def _f(val, default=None):
        try:
            v = float(val)
            return None if (v != v) else v  # NaN → None
        except (TypeError, ValueError):
            return default

    def _i(val, default=None):
        try:
            return int(float(val))
        except (TypeError, ValueError):
            return default

    def _s(val):
        v = str(val).strip() if val is not None else ""
        return v if v not in ("", "nan", "None") else None

    df = pd.read_csv(predictions_csv, dtype=str)
    date_slug = date_str.replace("-", "")
    rows = []

    for _, r in df.iterrows():
        home_display = _s(r.get("home")) or _s(r.get("home_cid", ""))
        away_display = _s(r.get("away")) or _s(r.get("away_cid", ""))
        game_id = f"bsb_{date_slug}_{_slugify(away_display)}_{_slugify(home_display)}"

        # Game time from schedule (start_utc column)
        start_utc = _s(r.get("start_utc"))
        game_time = start_utc if start_utc and start_utc != "nan" else None

        over_p = _f(r.get("over_prob"))

        data_blob = {
            "home_cid": _s(r.get("home_cid")),
            "away_cid": _s(r.get("away_cid")),
            "starters": {
                "home_starter": _s(r.get("home_starter")),
                "away_starter": _s(r.get("away_starter")),
                "hp_throws":    _s(r.get("hp_throws")),
                "ap_throws":    _s(r.get("ap_throws")),
                "hp_confirmed": bool(int(r.get("hp_confirmed", 0) or 0)),
                "ap_confirmed": bool(int(r.get("ap_confirmed", 0) or 0)),
            },
            "weather": {
                "temp_f":          _f(r.get("temp_f")),
                "wind_mph":        _f(r.get("wind_mph")),
                "wind_out_mph":    _f(r.get("wind_out_mph")),
                "weather_adj":     _f(r.get("weather_adj")),
                "wind_adj_raw":    _f(r.get("wind_adj_raw")),
                "weather_mode":    _s(r.get("weather_mode")),
                "weather_status":  _s(r.get("weather_status")),
                "rain_chance_pct": _f(r.get("rain_chance_pct")),
            },
            "park_factor":     _f(r.get("park_factor")),
            "bullpen": {
                "home_bullpen_adj": _f(r.get("home_bullpen_adj")),
                "away_bullpen_adj": _f(r.get("away_bullpen_adj")),
            },
            "home_rl_cover": _f(r.get("home_rl_cover")),
            "away_rl_cover": _f(r.get("away_rl_cover")),
        }

        rows.append({
            "sport":          "ncaa_baseball",
            "game_id":        game_id,
            "game_date":      date_str,
            "game_time":      game_time,
            "home_team":      home_display,
            "away_team":      away_display,
            "home_win_prob":  _f(r.get("home_win_prob")),
            "away_win_prob":  _f(r.get("away_win_prob")),
            "home_ml":        _i(r.get("ml_home")),
            "away_ml":        _i(r.get("ml_away")),
            "expected_total": _f(r.get("exp_total")),
            "model_spread":   round(_f(r.get("exp_home"), 0) - _f(r.get("exp_away"), 0), 4),
            "over_prob":      over_p,
            "under_prob":     round(1.0 - over_p, 4) if over_p is not None else None,
            "fragility":      _f(r.get("fragility_score")),
            "model_version":  "stan_v2_5000sim",
            "data":           json.dumps(data_blob),
        })

    if not rows:
        print("  Projections upload: 0 rows (empty dataframe)", file=sys.stderr)
        return 0

    # Build DSN without argparse namespace
    dsn = os.environ.get("DATABASE_URL", "")
    if not dsn:
        pw = os.environ.get("SUPABASE_DB_PASSWORD", "")
        if not pw:
            env_file = Path(__file__).parent.parent / ".env"
            if env_file.exists():
                for line in env_file.read_text().splitlines():
                    if line.startswith("SUPABASE_DB_PASSWORD="):
                        pw = line.split("=", 1)[1].strip().strip("'\"")
        if pw:
            dsn = DEFAULT_DSN.format(pw, PROJECT_REF)
    if not dsn:
        print("  Projections upload skipped — no DB credentials", file=sys.stderr)
        return 0

    with psycopg.connect(dsn, connect_timeout=30) as conn:
        with conn.cursor() as cur:
            sql = """
                INSERT INTO public.projections
                  (sport, game_id, game_date, game_time, home_team, away_team,
                   home_win_prob, away_win_prob, home_ml, away_ml,
                   expected_total, model_spread, over_prob, under_prob,
                   fragility, model_version, data, updated_at)
                VALUES
                  (%(sport)s, %(game_id)s, %(game_date)s, %(game_time)s,
                   %(home_team)s, %(away_team)s,
                   %(home_win_prob)s, %(away_win_prob)s, %(home_ml)s, %(away_ml)s,
                   %(expected_total)s, %(model_spread)s, %(over_prob)s, %(under_prob)s,
                   %(fragility)s, %(model_version)s, %(data)s::jsonb, now())
                ON CONFLICT (sport, game_id, model_version) DO UPDATE SET
                  home_win_prob  = EXCLUDED.home_win_prob,
                  away_win_prob  = EXCLUDED.away_win_prob,
                  home_ml        = EXCLUDED.home_ml,
                  away_ml        = EXCLUDED.away_ml,
                  expected_total = EXCLUDED.expected_total,
                  model_spread   = EXCLUDED.model_spread,
                  over_prob      = EXCLUDED.over_prob,
                  under_prob     = EXCLUDED.under_prob,
                  fragility      = EXCLUDED.fragility,
                  data           = EXCLUDED.data,
                  updated_at     = now()
            """
            cur.executemany(sql, rows)
        conn.commit()

    n = len(rows)
    print(f"  public.projections: {n} rows upserted", file=sys.stderr)
    return n


def load_predictions(cur, date_str: str):
    csv = Path(f"data/processed/predictions_{date_str}.csv")
    if not csv.exists():
        print(f"  Predictions not found: {csv}", file=sys.stderr)
        return 0
    df = pd.read_csv(csv, dtype=str)
    df["prediction_date"] = date_str
    # Rename columns
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
    return upsert_from_df(cur, df[keep], "predictions",
                          ["prediction_date", "home_canonical_id", "away_canonical_id"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Load data into Supabase baseball schema")
    parser.add_argument("--dsn", help="Postgres connection string")
    parser.add_argument("--all", action="store_true", help="Load all tables")
    parser.add_argument("--table", help="Load specific table")
    parser.add_argument("--date", help="Date for predictions (YYYY-MM-DD)")
    parser.add_argument("--fit-id", default="2026-03-21", help="Model fit ID")
    parser.add_argument("--season", type=int, default=2026)
    args = parser.parse_args()

    dsn = get_dsn(args)
    print(f"Connecting to database...", file=sys.stderr)

    with psycopg.connect(dsn, connect_timeout=30) as conn:
        with conn.cursor() as cur:
            if args.all:
                print("Loading all tables...", file=sys.stderr)
                load_teams(cur, args.season)
                load_stadiums(cur)
                load_games(cur)
                load_pitcher_appearances(cur)
                load_players(cur)
                load_run_events(cur)
                load_model_meta(cur, args.fit_id)
                load_team_params(cur, args.fit_id)
                load_pitcher_params(cur, args.fit_id)
                # Load all available prediction files
                for f in sorted(Path("data/processed").glob("predictions_2026-*.csv")):
                    date_str = f.stem.replace("predictions_", "").split("_")[0]
                    if len(date_str) == 10:  # YYYY-MM-DD
                        load_predictions(cur, date_str)
            elif args.table:
                t = args.table.lower()
                if t == "teams":
                    load_teams(cur, args.season)
                elif t == "stadiums":
                    load_stadiums(cur)
                elif t == "games":
                    load_games(cur)
                elif t in ("appearances", "pitcher_appearances"):
                    load_pitcher_appearances(cur)
                elif t == "players":
                    load_players(cur)
                elif t in ("run_events", "runevents"):
                    load_run_events(cur)
                elif t in ("model_meta", "meta"):
                    load_model_meta(cur, args.fit_id)
                elif t in ("team_params", "team_model_params"):
                    load_team_params(cur, args.fit_id)
                elif t in ("pitcher_params", "pitcher_model_params"):
                    load_pitcher_params(cur, args.fit_id)
                elif t == "predictions":
                    if not args.date:
                        print("--date required for predictions", file=sys.stderr)
                        return 1
                    load_predictions(cur, args.date)
                else:
                    print(f"Unknown table: {t}", file=sys.stderr)
                    return 1
            else:
                print("Specify --all or --table NAME", file=sys.stderr)
                return 1

            conn.commit()
            print("Done.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
