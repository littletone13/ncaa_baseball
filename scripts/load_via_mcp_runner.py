#!/usr/bin/env python3
"""Execute SQL batch files via stdout, designed to be piped to an MCP executor.

Combines multiple SQL files per table into larger batches and prints them
separated by a delimiter, so a wrapper can send each batch to execute_sql.

Usage:
  python3 scripts/load_via_mcp_runner.py --sql-dir tmp/sql_batches --table teams
  python3 scripts/load_via_mcp_runner.py --sql-dir tmp/sql_batches --all
"""
import argparse
import sys
from pathlib import Path

# Max chars per MCP execute_sql call (stay under ~60KB to be safe)
MAX_BATCH_CHARS = 55000
DELIMITER = "\n---MCP_BATCH_DELIMITER---\n"

TABLE_ORDER = [
    "teams", "stadiums", "games", "run_events",
    "pitcher_appearances", "players",
    "model_meta", "team_params", "pitcher_params",
    "predictions",
]


def combine_sql_files(sql_dir: Path, table: str) -> list[str]:
    """Combine SQL files for a table into batches under MAX_BATCH_CHARS."""
    files = sorted(sql_dir.glob(f"{table}_[0-9]*.sql"))
    if not files:
        return []

    batches = []
    current = ""
    for f in files:
        sql = f.read_text().strip()
        if current and len(current) + len(sql) + 2 > MAX_BATCH_CHARS:
            batches.append(current)
            current = sql
        else:
            current = current + "\n" + sql if current else sql
    if current:
        batches.append(current)

    return batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sql-dir", type=Path, default=Path("tmp/sql_batches"))
    parser.add_argument("--table", help="Specific table")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--count-only", action="store_true", help="Just count batches")
    args = parser.parse_args()

    tables = TABLE_ORDER if args.all else ([args.table] if args.table else [])
    if not tables:
        print("Specify --all or --table NAME", file=sys.stderr)
        return 1

    total = 0
    for table in tables:
        batches = combine_sql_files(args.sql_dir, table)
        if args.count_only:
            print(f"{table}: {len(batches)} combined batches", file=sys.stderr)
        else:
            for i, sql in enumerate(batches):
                print(f"-- TABLE: {table} BATCH: {i+1}/{len(batches)}", file=sys.stderr)
                print(sql)
                print(DELIMITER)
        total += len(batches)

    # Also handle prediction date files
    if args.all or args.table == "predictions":
        pred_files = sorted(args.sql_dir.glob("predictions_2026-*_*.sql"))
        pred_batches = []
        current = ""
        for f in pred_files:
            sql = f.read_text().strip()
            if current and len(current) + len(sql) + 2 > MAX_BATCH_CHARS:
                pred_batches.append(current)
                current = sql
            else:
                current = current + "\n" + sql if current else sql
        if current:
            pred_batches.append(current)

        if args.count_only:
            print(f"predictions (dated): {len(pred_batches)} combined batches", file=sys.stderr)
        else:
            for i, sql in enumerate(pred_batches):
                print(f"-- TABLE: predictions_dated BATCH: {i+1}/{len(pred_batches)}", file=sys.stderr)
                print(sql)
                print(DELIMITER)
        total += len(pred_batches)

    print(f"\nTotal: {total} MCP batches needed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
