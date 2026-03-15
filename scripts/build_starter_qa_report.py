"""Build a daily starter-resolution QA report from starters.csv."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0)


def build_starter_qa_report(
    starters_csv: Path,
    out_csv: Path,
    out_md: Path | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(starters_csv, dtype=str).fillna("")
    if df.empty:
        out = pd.DataFrame()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)
        return out

    # Build long format: one row per game-side starter.
    rows: list[dict[str, object]] = []
    for _, r in df.iterrows():
        for side in ("home", "away"):
            cid_col = "home_canonical_id" if "home_canonical_id" in df.columns else "home_cid"
            if side == "away":
                cid_col = "away_canonical_id" if "away_canonical_id" in df.columns else "away_cid"
            rows.append(
                {
                    "game_num": r.get("game_num", ""),
                    "side": side,
                    "team_cid": str(r.get(cid_col, "")).strip(),
                    "starter_name": r.get(f"{side}_starter", ""),
                    "starter_idx": _to_num(pd.Series([r.get(f"{side}_starter_idx", "")])).iloc[0],
                    "resolution_method": r.get(f"{side}_resolution_method", ""),
                    "d1b_fallback": _to_num(pd.Series([r.get(f"{side}_d1b_fallback", "0")])).iloc[0],
                }
            )

    long_df = pd.DataFrame(rows)
    long_df["idx_gt_0"] = (long_df["starter_idx"] > 0).astype(int)

    miss_board = (
        long_df.groupby("team_cid", dropna=False)["idx_gt_0"]
        .agg(starters="count", idx_gt_0="sum")
        .reset_index()
    )
    miss_board["idx0_count"] = miss_board["starters"] - miss_board["idx_gt_0"]
    miss_board["idx0_rate"] = (miss_board["idx0_count"] / miss_board["starters"]).round(4)
    miss_board = miss_board.sort_values(["idx0_rate", "idx0_count"], ascending=[False, False])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    miss_board.to_csv(out_csv, index=False)

    if out_md is not None:
        total = len(long_df)
        pct_idx = float(long_df["idx_gt_0"].mean() * 100.0) if total else 0.0
        pct_id = float((long_df["resolution_method"] == "id_match").mean() * 100.0) if total else 0.0
        pct_name = float((long_df["resolution_method"] == "name_match").mean() * 100.0) if total else 0.0
        pct_lookup = float((long_df["resolution_method"] == "lookup_only").mean() * 100.0) if total else 0.0
        pct_d1b = float((long_df["d1b_fallback"] > 0).mean() * 100.0) if total else 0.0
        top = miss_board.head(15)
        lines = [
            "# Starter Resolution QA",
            "",
            f"- Starter slots: {total}",
            f"- `idx > 0` coverage: {pct_idx:.1f}%",
            f"- Resolved by ID: {pct_id:.1f}%",
            f"- Resolved by name: {pct_name:.1f}%",
            f"- Lookup-only (no table match): {pct_lookup:.1f}%",
            f"- D1B fallback share: {pct_d1b:.1f}%",
            "",
            "## Team Miss Leaderboard (`idx == 0`)",
            "",
            "| team_cid | starters | idx0_count | idx0_rate |",
            "|---|---:|---:|---:|",
        ]
        for _, r in top.iterrows():
            lines.append(
                f"| {r['team_cid']} | {int(r['starters'])} | {int(r['idx0_count'])} | {float(r['idx0_rate']):.2%} |"
            )
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("\n".join(lines), encoding="utf-8")

    return miss_board


def main() -> int:
    parser = argparse.ArgumentParser(description="Build daily starter QA report.")
    parser.add_argument("--starters", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()
    build_starter_qa_report(args.starters, args.out_csv, args.out_md)
    print(f"Wrote starter QA report: {args.out_csv}")
    if args.out_md:
        print(f"Wrote starter QA markdown: {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
