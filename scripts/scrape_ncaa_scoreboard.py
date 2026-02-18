from __future__ import annotations

import argparse
import csv
import dataclasses
import html as htmllib
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Final


@dataclass(frozen=True, slots=True)
class Game:
    academic_year: int
    division: int
    sport_code: str
    game_date: str  # YYYY-MM-DD
    contest_id: int
    away_team_ncaa_id: int
    away_team_name: str
    home_team_ncaa_id: int
    home_team_name: str
    away_runs: int | None
    home_runs: int | None
    neutral_site: str | None
    attendance: int | None


_GAME_START_RE = re.compile(
    r"<tr>\s*<td\s+rowspan=\"2\"[^>]*>(?P<date>[^<]+)</td>",
    re.IGNORECASE | re.DOTALL,
)
_GAME_BORDER_RE = re.compile(r'<tr\s+style="border-bottom:\s*1px\s+solid\s+#cccccc"', re.IGNORECASE)
_CONTEST_RE = re.compile(r'href="/contests/(?P<contest>\d+)/box_score"', re.IGNORECASE)
_TEAM_LINK_RE = re.compile(r'href="/teams/(?P<id>\d+)">\s*(?P<name>[^<]+)</a>', re.IGNORECASE)
_TOTALCOL_RE = re.compile(r'class="totalcol"\s*>\s*(?P<runs>\d+)', re.IGNORECASE)
_NEUTRAL_SITE_RE = re.compile(r'<td\s+rowspan="2"\s+valign="center">\s*(?P<txt>.*?)</td>', re.IGNORECASE | re.DOTALL)
_ATTENDANCE_RE = re.compile(
    r'<td\s+rowspan="2"\s+align="right"\s+valign="center">\s*(?P<num>\d+)\s*</td>',
    re.IGNORECASE,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


_CURL_BIN: Final[str] = "curl"


def _looks_blocked(html: str) -> bool:
    head = html[:2000].lower()
    return (
        "<title>access denied</title>" in head
        or "access denied" in head and "errors.edgesuite.net" in head
        or "akamai_validation" in head
        or "bm-verify" in head
    )


def _curl(url: str, *, max_time_s: int, retries: int, user_agent: str | None) -> str:
    """
    Fetch via curl with basic resilience.

    - Uses curl's own retry flags for transient network issues
    - Also retries the whole command to handle timeouts (exit 28)
    """
    cmd = [
        _CURL_BIN,
        "-sL",
        "--compressed",
        "--max-time",
        str(max_time_s),
        "--retry",
        "3",
        "--retry-delay",
        "1",
        "--retry-all-errors",
    ]
    if user_agent:
        cmd += ["-H", f"User-Agent: {user_agent}"]
    cmd.append(url)
    last_err: subprocess.CalledProcessError | None = None
    for _ in range(retries):
        try:
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return proc.stdout
        except subprocess.CalledProcessError as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    raise RuntimeError("curl failed unexpectedly")


def _parse_game_date(raw: str, academic_year: int) -> str:
    # Examples: "02/16/2025 TBA" or "02/16/2025 1:00 PM"
    raw = htmllib.unescape(raw).strip()
    mmddyyyy = raw.split()[0]
    dt = datetime.strptime(mmddyyyy, "%m/%d/%Y").date()
    if dt.year != academic_year and dt.year not in (academic_year, academic_year - 1):
        # NCAA "academic_year" can contain games in the adjacent calendar year; keep explicit parsed date.
        pass
    return dt.isoformat()


def _clean_team_name(s: str) -> str:
    s = htmllib.unescape(s).strip()
    # Strip ranking prefix: "#1 Team" or "No. 1 Team"
    s = re.sub(r"^\s*(#|No\.)\s*\d+\s+", "", s, flags=re.IGNORECASE)
    # Strip record suffix: "Team (3-2)" or "Team (0-0)"
    # Some records include ties: "(24-26-1)"
    s = re.sub(r"\s*\(\d+-\d+(?:-\d+)?\)\s*$", "", s)
    return " ".join(s.split())


def _extract_text_fragment(s: str) -> str:
    s = htmllib.unescape(s)
    # Remove tags and collapse whitespace.
    s = re.sub(r"<[^>]+>", " ", s)
    s = " ".join(s.split())
    return s.strip() or ""


def parse_games(html: str, academic_year: int, division: int, sport_code: str) -> list[Game]:
    games: list[Game] = []
    starts = list(_GAME_START_RE.finditer(html))
    for i, m in enumerate(starts):
        raw_date = m.group("date")
        start_pos = m.start()

        # Each game block ends at the next "border-bottom" separator row; some games have no box score.
        next_border = _GAME_BORDER_RE.search(html, m.end())
        if next_border is None:
            end_pos = starts[i + 1].start() if i + 1 < len(starts) else len(html)
        else:
            tr_end = html.find("</tr>", next_border.start())
            end_pos = (tr_end + 5) if tr_end != -1 else next_border.end()

        chunk = html[start_pos:end_pos]
        cm = _CONTEST_RE.search(chunk)
        if not cm:
            # Canceled/exhibition or otherwise unlinked games: skip (no deterministic contest_id).
            continue
        contest_id = int(cm.group("contest"))

        team_links = list(_TEAM_LINK_RE.finditer(chunk))
        if len(team_links) < 2:
            continue
        away = team_links[0]
        home = team_links[1]

        away_id = int(away.group("id"))
        home_id = int(home.group("id"))
        away_name = _clean_team_name(away.group("name"))
        home_name = _clean_team_name(home.group("name"))

        totals = [int(x.group("runs")) for x in _TOTALCOL_RE.finditer(chunk)]
        away_runs = totals[0] if len(totals) >= 1 else None
        home_runs = totals[1] if len(totals) >= 2 else None

        neutral_site = None
        nsm = _NEUTRAL_SITE_RE.search(chunk)
        if nsm:
            txt = _extract_text_fragment(nsm.group("txt"))
            neutral_site = txt or None

        attendance = None
        am = _ATTENDANCE_RE.search(chunk)
        if am:
            attendance = int(am.group("num"))

        games.append(
            Game(
                academic_year=academic_year,
                division=division,
                sport_code=sport_code,
                game_date=_parse_game_date(raw_date, academic_year),
                contest_id=contest_id,
                away_team_ncaa_id=away_id,
                away_team_name=away_name,
                home_team_ncaa_id=home_id,
                home_team_name=home_name,
                away_runs=away_runs,
                home_runs=home_runs,
                neutral_site=neutral_site,
                attendance=attendance,
            )
        )
    return games


def _iter_dates(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("end must be >= start")
    out: list[date] = []
    d = start
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out


def _write_csv(path: Path, games: list[Game]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "academic_year",
        "division",
        "sport_code",
        "game_date",
        "contest_id",
        "away_team_ncaa_id",
        "away_team_name",
        "home_team_ncaa_id",
        "home_team_name",
        "away_runs",
        "home_runs",
        "neutral_site",
        "attendance",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for g in games:
            w.writerow(dataclasses.asdict(g))


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape NCAA scoreboard (schedule_list) into a clean games table.")
    parser.add_argument("--academic-year", type=int, required=True, help="Academic year (YYYY), e.g. 2026")
    parser.add_argument("--division", type=int, default=1, help="Division (default: 1)")
    parser.add_argument("--sport-code", default="MBA", help="Sport code (baseball: MBA)")
    parser.add_argument("--conf-id", type=int, default=-1, help="Conference id (-1 for all teams)")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/ncaa/scoreboard"), help="Raw HTML output dir")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: data/processed/scoreboard/games_<year>.csv)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite cached raw HTML files")
    parser.add_argument("--max-time", type=int, default=45, help="curl --max-time seconds (default: 45)")
    parser.add_argument("--retries", type=int, default=4, help="Fetch retries per day (default: 4)")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests (default: 0)")
    parser.add_argument("--user-agent", default=None, help="Custom User-Agent header (optional)")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    out = args.out or Path("data/processed/scoreboard") / f"games_{args.academic_year}.csv"

    all_games: dict[int, Game] = {}
    failures: list[dict[str, str]] = []
    for d in _iter_dates(start, end):
        mmddyyyy = d.strftime("%m/%d/%Y")
        url = (
            "https://stats.ncaa.org/team/schedule_list"
            f"?academic_year={args.academic_year}"
            f"&conf_id={args.conf_id}"
            f"&division={args.division}"
            f"&sport_code={args.sport_code}"
            f"&game_date={mmddyyyy}"
        )

        raw_path = args.raw_dir / f"academic_year={args.academic_year}" / f"{d.isoformat()}.html"
        if raw_path.exists() and not args.overwrite:
            html = raw_path.read_text(encoding="utf-8", errors="replace")
        else:
            try:
                html = _curl(url, max_time_s=args.max_time, retries=args.retries, user_agent=args.user_agent)
                if _looks_blocked(html):
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    denied_path = raw_path.with_suffix(".denied.html")
                    denied_path.write_text(html, encoding="utf-8")
                    err_path = raw_path.with_suffix(".error.txt")
                    err_path.write_text(
                        f"fetched_at={_utc_now_iso()}\nurl={url}\nblocked=1\n",
                        encoding="utf-8",
                    )
                    failures.append(
                        {
                            "academic_year": str(args.academic_year),
                            "game_date": d.isoformat(),
                            "url": url,
                            "exit_code": "blocked",
                        }
                    )
                    continue
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text(html, encoding="utf-8")
            except subprocess.CalledProcessError as e:
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                err_path = raw_path.with_suffix(".error.txt")
                err_path.write_text(
                    f"fetched_at={_utc_now_iso()}\nurl={url}\nexit_code={e.returncode}\n",
                    encoding="utf-8",
                )
                failures.append(
                    {
                        "academic_year": str(args.academic_year),
                        "game_date": d.isoformat(),
                        "url": url,
                        "exit_code": str(e.returncode),
                    }
                )
                continue
            finally:
                if args.sleep and args.sleep > 0:
                    time.sleep(args.sleep)

        games = parse_games(html, args.academic_year, args.division, args.sport_code)
        for g in games:
            # Deduplicate by contest_id; last write wins (should be identical).
            all_games[g.contest_id] = g

    _write_csv(out, list(all_games.values()))
    print(f"Wrote {len(all_games)} unique contests -> {out}")
    print(f"Raw HTML cached under {args.raw_dir}/academic_year={args.academic_year}/ (fetched_at={_utc_now_iso()})")
    if failures:
        fail_path = out.with_suffix(".failures.csv")
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        with fail_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["academic_year", "game_date", "url", "exit_code"])
            w.writeheader()
            w.writerows(failures)
        print(f"WARNING: {len(failures)} days failed -> {fail_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
