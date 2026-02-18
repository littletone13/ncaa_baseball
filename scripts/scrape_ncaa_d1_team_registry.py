from __future__ import annotations

import argparse
import csv
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final


_CURL_BIN: Final[str] = "curl"


@dataclass(frozen=True, slots=True)
class Conference:
    conference_id: int
    conference: str


@dataclass(frozen=True, slots=True)
class TeamRow:
    academic_year: int
    division: int
    sport_code: str
    conference_id: int
    conference: str
    ncaa_teams_id: int
    team_name: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _looks_blocked(html: str) -> bool:
    head = html[:2000].lower()
    return (
        "<title>access denied</title>" in head
        or "errors.edgesuite.net" in head
        or "akamai_validation" in head
        or "bm-verify" in head
    )


def _curl(url: str, *, max_time_s: int, user_agent: str | None) -> str:
    cmd = [_CURL_BIN, "-sL", "--compressed", "--max-time", str(max_time_s)]
    if user_agent:
        cmd += ["-H", f"User-Agent: {user_agent}"]
    cmd.append(url)
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return proc.stdout


_CONF_RE = re.compile(r'changeConference\((?P<id>-?\d+)\);">(?P<name>[^<]+)</a>')
_TEAM_RE = re.compile(r'<a href="/teams/(?P<id>\d+)">(?P<name>[^<]+)</a>')


def _parse_conferences(html: str) -> list[Conference]:
    confs: dict[int, str] = {}
    for m in _CONF_RE.finditer(html):
        cid = int(m.group("id"))
        name = m.group("name").strip()
        if cid == -1:
            continue
        confs[cid] = name
    return [Conference(conference_id=k, conference=confs[k]) for k in sorted(confs)]


def _parse_teams(html: str) -> list[tuple[int, str]]:
    teams: dict[int, str] = {}
    for m in _TEAM_RE.finditer(html):
        tid = int(m.group("id"))
        name = m.group("name").strip()
        teams[tid] = name
    return [(k, teams[k]) for k in sorted(teams)]


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_yaml(path: Path, academic_year: int, division: int, sport_code: str, teams: list[TeamRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')

    with path.open("w", encoding="utf-8") as f:
        f.write("# Auto-generated from stats.ncaa.org team list\n")
        f.write(f"# Generated at: {_utc_now_iso()}\n")
        f.write("meta:\n")
        f.write(f"  academic_year: {academic_year}\n")
        f.write(f"  division: {division}\n")
        f.write(f"  sport_code: {sport_code}\n")
        f.write('  source_url: "https://stats.ncaa.org/team/inst_team_list"\n')
        f.write("teams:\n")
        for t in teams:
            # Use NCAA's team-season id as the primary identifier (no name-based slugging).
            f.write(f'  - id: "NCAA_{t.ncaa_teams_id}"\n')
            f.write(f"    ncaa_teams_id: {t.ncaa_teams_id}\n")
            f.write(f'    school: "{esc(t.team_name)}"\n')
            f.write(f'    conference: "{esc(t.conference)}"\n')
            f.write(f"    conference_id: {t.conference_id}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scrape NCAA D1 baseball team registry (no fuzzy matching) from stats.ncaa.org."
    )
    parser.add_argument("--academic-year", type=int, default=2026, help="Academic year (YYYY), e.g. 2026")
    parser.add_argument("--division", type=int, default=1, help="NCAA division (default: 1)")
    parser.add_argument("--sport-code", default="MBA", help="NCAA sport_code (baseball: MBA)")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/ncaa/inst_team_list"), help="Raw HTML cache dir")
    parser.add_argument("--max-time", type=int, default=30, help="curl --max-time seconds (default: 30)")
    parser.add_argument("--user-agent", default=None, help="Custom User-Agent header (optional)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite cached raw HTML files")
    parser.add_argument("--min-teams", type=int, default=250, help="Fail if parsed teams are below this count")
    parser.add_argument("--allow-partial", action="store_true", help="Allow writing outputs even if below --min-teams")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: data/registries/ncaa_d1_teams_<year>.csv)",
    )
    parser.add_argument(
        "--out-conferences-csv",
        type=Path,
        default=None,
        help="Output conferences CSV path (default: data/registries/ncaa_conferences_<year>.csv)",
    )
    parser.add_argument(
        "--out-yaml",
        type=Path,
        default=None,
        help="Output YAML path (default: teams_baseball_d1_<year>.yaml)",
    )
    args = parser.parse_args()

    out_csv = args.out_csv or Path("data/registries") / f"ncaa_d1_teams_{args.academic_year}.csv"
    out_conf_csv = args.out_conferences_csv or Path("data/registries") / f"ncaa_conferences_{args.academic_year}.csv"
    out_yaml = args.out_yaml or Path(f"teams_baseball_d1_{args.academic_year}.yaml")

    base = (
        "https://stats.ncaa.org/team/inst_team_list"
        f"?academic_year={args.academic_year}"
        f"&conf_id=-1"
        f"&division={args.division}"
        f"&sport_code={args.sport_code}"
    )
    base_cache = args.raw_dir / f"academic_year={args.academic_year}" / "conf_id=-1.html"
    if base_cache.exists() and not args.overwrite:
        html = base_cache.read_text(encoding="utf-8", errors="replace")
    else:
        html = _curl(base, max_time_s=args.max_time, user_agent=args.user_agent)
        base_cache.parent.mkdir(parents=True, exist_ok=True)
        base_cache.write_text(html, encoding="utf-8")

    if _looks_blocked(html):
        raise SystemExit(
            f"Blocked by stats.ncaa.org (Access Denied / interstitial). Cached: {base_cache}. "
            "Try again later / different network, or switch to a browser-based workflow."
        )
    conferences = _parse_conferences(html)
    if not conferences:
        raise SystemExit("Failed to parse conferences from NCAA page (site markup changed?).")

    conf_rows = [
        {
            "academic_year": args.academic_year,
            "division": args.division,
            "sport_code": args.sport_code,
            "conference_id": c.conference_id,
            "conference": c.conference,
        }
        for c in conferences
    ]
    _write_csv(
        out_conf_csv,
        conf_rows,
        ["academic_year", "division", "sport_code", "conference_id", "conference"],
    )

    teams_by_id: dict[int, TeamRow] = {}
    conflicts: list[tuple[int, str, str]] = []

    for c in conferences:
        url = (
            "https://stats.ncaa.org/team/inst_team_list"
            f"?academic_year={args.academic_year}"
            f"&conf_id={c.conference_id}"
            f"&division={args.division}"
            f"&sport_code={args.sport_code}"
        )
        conf_cache = args.raw_dir / f"academic_year={args.academic_year}" / f"conf_id={c.conference_id}.html"
        if conf_cache.exists() and not args.overwrite:
            conf_html = conf_cache.read_text(encoding="utf-8", errors="replace")
        else:
            try:
                conf_html = _curl(url, max_time_s=args.max_time, user_agent=args.user_agent)
            except subprocess.CalledProcessError:
                continue
            conf_cache.parent.mkdir(parents=True, exist_ok=True)
            conf_cache.write_text(conf_html, encoding="utf-8")

        if _looks_blocked(conf_html):
            continue
        for team_id, team_name in _parse_teams(conf_html):
            row = TeamRow(
                academic_year=args.academic_year,
                division=args.division,
                sport_code=args.sport_code,
                conference_id=c.conference_id,
                conference=c.conference,
                ncaa_teams_id=team_id,
                team_name=team_name,
            )
            prev = teams_by_id.get(team_id)
            if prev is not None and (prev.conference_id != row.conference_id or prev.team_name != row.team_name):
                conflicts.append((team_id, f"{prev.team_name} ({prev.conference})", f"{row.team_name} ({row.conference})"))
                continue
            teams_by_id[team_id] = row

    if conflicts:
        msg = "\n".join([f"- {tid}: {a} vs {b}" for tid, a, b in conflicts[:50]])
        raise SystemExit(f"Conference assignment conflicts detected (first 50 shown):\n{msg}")

    teams = [teams_by_id[k] for k in sorted(teams_by_id)]
    if len(teams) < args.min_teams and not args.allow_partial:
        raise SystemExit(
            f"Parsed only {len(teams)} teams (< {args.min_teams}). "
            "This usually means scraping was blocked or markup changed. "
            f"Re-run with --allow-partial to write anyway, or inspect cache under {args.raw_dir}."
        )
    team_rows = [
        {
            "academic_year": t.academic_year,
            "division": t.division,
            "sport_code": t.sport_code,
            "conference_id": t.conference_id,
            "conference": t.conference,
            "ncaa_teams_id": t.ncaa_teams_id,
            "team_name": t.team_name,
        }
        for t in teams
    ]
    _write_csv(
        out_csv,
        team_rows,
        ["academic_year", "division", "sport_code", "conference_id", "conference", "ncaa_teams_id", "team_name"],
    )

    teams_sorted_for_yaml = sorted(teams, key=lambda x: (x.conference, x.team_name, x.ncaa_teams_id))
    _write_yaml(out_yaml, args.academic_year, args.division, args.sport_code, teams_sorted_for_yaml)

    print(f"Wrote {len(conferences)} conferences -> {out_conf_csv}")
    print(f"Wrote {len(teams)} teams -> {out_csv}")
    print(f"Wrote YAML -> {out_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
