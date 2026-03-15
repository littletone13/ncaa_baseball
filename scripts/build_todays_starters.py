"""
Build today's NCAA D1 starter table for projections/simulation.

Source priority (per side, per game):
1) Manual starter file (announced from D1Baseball/team sites)
2) D1Baseball same-day lineup table (if available)
3) ESPN summary boxscore starter (in-progress / just-started games)
4) Rotation inference from historical ESPN starts (pitching_lines_espn.csv)
5) Unknown (graceful fallback; never crash)

Output is keyed by game (event_id + teams + date) and includes:
- canonical team IDs
- pitcher ESPN IDs / names
- run-event pitcher/team indices where available
- source + confidence fields for auditability

Usage:
  python3 scripts/build_todays_starters.py --date 2026-02-25 --write-manual-template
  # Fill manual starter rows from public sources, then run again:
  python3 scripts/build_todays_starters.py --date 2026-02-25
"""
from __future__ import annotations

import argparse
import html
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import pandas as pd
import requests
from bs4 import BeautifulSoup

from ncaa_baseball.phase1 import (
    build_odds_name_to_canonical,
    load_canonical_teams,
    resolve_odds_teams,
)

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/baseball/college-baseball"
UA = "Mozilla/5.0 (X11; Linux x86_64)"
FINAL_STATUSES = {"STATUS_FINAL", "STATUS_FULL_TIME"}
SCHEDULED_STATUSES = {
    "STATUS_SCHEDULED",
    "STATUS_CREATED",
    "STATUS_POSTPONED",
    "STATUS_CANCELED",
    "STATUS_DELAYED",
}

_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")
_SPACE_RE = re.compile(r"\s+")
_GAME_DATE_RE = re.compile(r"\b([A-Z][a-z]{2}),\s*([A-Z][a-z]{2})\s+(\d{1,2})\b")
_OPP_RE = re.compile(r"\b(?:vs\.?|at)\s+(.+?)(?:\s*\(|$)", re.IGNORECASE)


@dataclass
class StarterPick:
    espn_id: str
    name: str
    source: str
    confidence: float
    note: str


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = html.unescape(str(s))
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().replace("&", " and ")
    s = s.replace("’", "'")
    s = s.replace("st.", "st")
    s = _PUNCT_RE.sub(" ", s)
    s = _SPACE_RE.sub(" ", s).strip()
    return s


def normalize_team_name(s: str) -> str:
    s = normalize_text(s)
    # Remove common mascot-ish suffix words when they appear at the end.
    suffix_words = {
        "bulldogs", "wildcats", "tigers", "razorbacks", "longhorns", "aggies",
        "cardinals", "trojans", "warriors", "spartans", "panthers", "bears",
        "knights", "titans", "huskies", "hawks", "eagles", "owls", "pirates",
        "rams", "lions", "bobcats", "mustangs", "rebels", "gators", "seminoles",
        "volunteers", "bluejays", "dons", "broncos", "vikings", "blazers",
        "mountaineers", "colonials", "cougars", "gaels", "terriers", "bearcats",
    }
    parts = s.split()
    if len(parts) >= 2 and parts[-1] in suffix_words:
        s = " ".join(parts[:-1])
    return s


def normalize_person_name(s: str) -> str:
    s = normalize_text(s)
    s = s.replace(" jr", "").replace(" sr", "")
    s = s.replace(" iii", "").replace(" ii", "")
    return _SPACE_RE.sub(" ", s).strip()


def team_names_match(a: str, b: str) -> bool:
    na = normalize_team_name(a)
    nb = normalize_team_name(b)
    if not na or not nb:
        return False
    return na == nb or na.startswith(nb + " ") or nb.startswith(na + " ")


def _team_forms_from_norm(norm_name: str) -> set[str]:
    tokens = norm_name.split()
    forms = {norm_name}
    if not tokens:
        return forms
    token_alts: dict[str, list[str]] = {
        "st": ["state", "saint"],
        "state": ["st"],
        "saint": ["st"],
        "fla": ["florida"],
        "ga": ["georgia"],
        "ky": ["kentucky"],
        "intl": ["international"],
        "ft": ["fort"],
        "mt": ["mount"],
    }
    acronym_alts: dict[str, list[str]] = {
        "fiu": ["florida international"],
        "fgcu": ["florida gulf coast"],
        "ucf": ["central florida"],
        "uic": ["illinois chicago"],
        "umbc": ["maryland baltimore county"],
        "umes": ["maryland eastern shore"],
        "uncw": ["north carolina wilmington"],
        "utsa": ["texas san antonio"],
        "utrgv": ["texas rio grande valley"],
        "dbu": ["dallas baptist"],
        "etsu": ["east tennessee state"],
        "siue": ["southern illinois edwardsville"],
        "sfa": ["stephen f austin"],
        "csun": ["cal state northridge"],
        "liu": ["long island"],
        "njit": ["new jersey institute of technology"],
        "usc": ["south carolina", "southern california"],
    }
    for i, tok in enumerate(tokens):
        for alt in token_alts.get(tok, []):
            forms.add(" ".join(tokens[:i] + alt.split() + tokens[i + 1:]))
        for alt in acronym_alts.get(tok, []):
            forms.add(" ".join(tokens[:i] + alt.split() + tokens[i + 1:]))
    return {f for f in forms if f}


def build_canonical_name_index(canonical: pd.DataFrame) -> dict[str, list[tuple[str, int, str]]]:
    idx: dict[str, list[tuple[str, int, str]]] = {}
    for _, row in canonical.iterrows():
        cid = str(row.get("canonical_id") or "").strip()
        if not cid:
            continue
        tname = str(row.get("team_name") or "").strip()
        try:
            tid = int(row.get("ncaa_teams_id"))
        except Exception:
            continue
        for form in _team_forms_from_norm(normalize_team_name(tname)):
            idx.setdefault(form, []).append((cid, tid, tname))
    return idx


def resolve_team_name_to_canonical(
    team_name: str,
    canonical: pd.DataFrame,
    name_to_canonical: dict[str, tuple[str, int]],
    canonical_name_index: dict[str, list[tuple[str, int, str]]],
) -> tuple[str, int] | None:
    n = normalize_team_name(team_name)
    if n:
        cand: list[tuple[str, int, str]] = []
        for form in _team_forms_from_norm(n):
            cand.extend(canonical_name_index.get(form, []))
        # De-duplicate by canonical id
        by_cid: dict[str, tuple[str, int, str]] = {}
        for c in cand:
            by_cid[c[0]] = c
        cand = list(by_cid.values())
        if len(cand) == 1:
            c = cand[0]
            return (c[0], c[1])
        if len(cand) > 1:
            # Prefer the longest matching canonical school name.
            c = sorted(cand, key=lambda x: -len(normalize_team_name(x[2])))[0]
            return (c[0], c[1])

        # If ESPN appends mascot words, peel trailing tokens and retry.
        toks = n.split()
        while len(toks) > 1:
            toks = toks[:-1]
            n_trim = " ".join(toks)
            cand_trim: list[tuple[str, int, str]] = []
            for form in _team_forms_from_norm(n_trim):
                cand_trim.extend(canonical_name_index.get(form, []))
            by_cid_trim: dict[str, tuple[str, int, str]] = {}
            for c in cand_trim:
                by_cid_trim[c[0]] = c
            cand_trim = list(by_cid_trim.values())
            if len(cand_trim) == 1:
                c = cand_trim[0]
                return (c[0], c[1])

    # Fallback to existing odds-style resolver, but reject prefix-only over-matches
    # (e.g. "Florida International" incorrectly mapping to "Florida").
    t, _ = resolve_odds_teams(team_name, team_name, canonical, name_to_canonical)
    if not t:
        return None
    cid = str(t[0])
    row = canonical.loc[canonical["canonical_id"].astype(str) == cid]
    if row.empty:
        return t
    c_name = str(row.iloc[0].get("team_name") or "").strip()
    c_norm = normalize_team_name(c_name)
    if c_norm and n.startswith(c_norm + " ") and len(n.split()) > len(c_norm.split()):
        return None
    return t


def fetch_json(session: requests.Session, url: str, *, retries: int = 3, timeout: int = 20) -> dict | None:
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=timeout, headers={"User-Agent": UA})
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt == retries - 1:
                return None
    return None


def parse_date_str(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def load_manual_starters(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, dtype=str).fillna("")
    for c in (
        "game_date",
        "event_id",
        "home_team",
        "away_team",
        "home_canonical_id",
        "away_canonical_id",
        "home_pitcher_name",
        "away_pitcher_name",
        "home_pitcher_espn_id",
        "away_pitcher_espn_id",
        "source",
        "source_url",
        "notes",
    ):
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].astype(str).str.strip()
    return df


def find_manual_row(game: dict[str, Any], manual_df: pd.DataFrame) -> pd.Series | None:
    if manual_df.empty:
        return None
    event_id = str(game.get("event_id") or "").strip()
    if event_id:
        m = manual_df.loc[manual_df["event_id"].astype(str).str.strip() == event_id]
        if not m.empty:
            return m.iloc[0]

    gdate = str(game.get("game_date") or "").strip()
    subset = manual_df
    if gdate:
        subset = subset.loc[(subset["game_date"] == "") | (subset["game_date"] == gdate)]
    if subset.empty:
        return None
    for _, row in subset.iterrows():
        if row.get("home_canonical_id") and game.get("home_canonical_id"):
            home_ok = row["home_canonical_id"] == game["home_canonical_id"]
        else:
            home_ok = team_names_match(row.get("home_team", ""), game.get("home_team_name", ""))
        if row.get("away_canonical_id") and game.get("away_canonical_id"):
            away_ok = row["away_canonical_id"] == game["away_canonical_id"]
        else:
            away_ok = team_names_match(row.get("away_team", ""), game.get("away_team_name", ""))
        if home_ok and away_ok:
            return row
    return None


def extract_starters_from_summary(summary: dict, home_team_id: str, away_team_id: str) -> dict[str, dict[str, str]]:
    out = {"home": {"id": "", "name": ""}, "away": {"id": "", "name": ""}}
    players = (((summary or {}).get("boxscore") or {}).get("players") or [])
    for team_section in players:
        team_id = str((team_section.get("team") or {}).get("id") or "")
        for stat_cat in (team_section.get("statistics") or []):
            labels = stat_cat.get("labels") or []
            if "IP" not in labels:
                continue
            for athlete in (stat_cat.get("athletes") or []):
                if not athlete.get("starter"):
                    continue
                ath = athlete.get("athlete") or {}
                pid = str(ath.get("id") or "").strip()
                pname = str(ath.get("displayName") or "").strip()
                if team_id == str(home_team_id):
                    out["home"] = {"id": pid, "name": pname}
                elif team_id == str(away_team_id):
                    out["away"] = {"id": pid, "name": pname}
    return out


def load_pitching_starts(path: Path) -> pd.DataFrame:
    cols = [
        "game_date", "canonical_id", "team_name", "pitcher_espn_id", "pitcher_name", "starter",
    ]
    if not path.exists():
        return pd.DataFrame(columns=cols + ["team_key", "pitcher_name_norm"])
    df = pd.read_csv(path, dtype=str).fillna("")
    missing = [c for c in cols if c not in df.columns]
    for c in missing:
        df[c] = ""
    starter_mask = df["starter"].astype(str).str.lower().isin({"1", "true", "t", "yes"})
    df = df.loc[starter_mask].copy()
    if df.empty:
        return pd.DataFrame(columns=cols + ["team_key", "pitcher_name_norm"])
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    df = df.loc[df["game_date"].notna()].copy()
    df["pitcher_espn_id"] = df["pitcher_espn_id"].astype(str).str.strip()
    df["pitcher_name"] = df["pitcher_name"].astype(str).str.strip()
    df["canonical_id"] = df["canonical_id"].astype(str).str.strip()
    df["team_name"] = df["team_name"].astype(str).str.strip()
    df["team_key"] = df["canonical_id"].where(df["canonical_id"] != "", df["team_name"].map(normalize_team_name))
    df["pitcher_name_norm"] = df["pitcher_name"].map(normalize_person_name)
    df = df.loc[df["team_key"] != ""].copy()
    return df


def resolve_name_to_pitcher_id(
    team_key: str,
    pitcher_name: str,
    starts: pd.DataFrame,
) -> tuple[str, str]:
    """Return (pitcher_espn_id, canonical_name) if found."""
    if starts.empty or not team_key or not pitcher_name:
        return ("", "")
    name_norm = normalize_person_name(pitcher_name)
    if not name_norm:
        return ("", "")
    team_df = starts.loc[starts["team_key"] == team_key]
    if team_df.empty:
        return ("", "")

    exact = team_df.loc[team_df["pitcher_name_norm"] == name_norm]
    if not exact.empty:
        r = exact.sort_values("game_date").iloc[-1]
        return (str(r["pitcher_espn_id"]).strip(), str(r["pitcher_name"]).strip())

    # Fallback: last-name + first initial
    tokens = name_norm.split()
    if not tokens:
        return ("", "")
    last = tokens[-1]
    first_initial = tokens[0][0]
    cand = team_df.loc[
        team_df["pitcher_name_norm"].str.split().str[-1].fillna("") == last
    ].copy()
    if cand.empty:
        return ("", "")
    cand = cand.loc[cand["pitcher_name_norm"].str[0] == first_initial]
    if cand.empty:
        return ("", "")
    r = cand.sort_values("game_date").iloc[-1]
    return (str(r["pitcher_espn_id"]).strip(), str(r["pitcher_name"]).strip())


def infer_probable_starter(team_key: str, game_day: date, starts: pd.DataFrame) -> StarterPick | None:
    if starts.empty or not team_key:
        return None
    tdf = starts.loc[(starts["team_key"] == team_key) & (starts["pitcher_espn_id"] != "")]
    if tdf.empty:
        return None
    tdf = tdf.loc[tdf["game_date"] < game_day].copy()
    if tdf.empty:
        return None

    target_rest = 7 if game_day.weekday() in (4, 5, 6) else 4
    rows: list[dict[str, Any]] = []
    for pid, grp in tdf.groupby("pitcher_espn_id"):
        grp = grp.sort_values("game_date")
        last_row = grp.iloc[-1]
        last_date = last_row["game_date"]
        if not isinstance(last_date, date):
            continue
        days_since = (game_day - last_date).days
        if days_since <= 0:
            continue
        starts_n = int(len(grp))
        start_dates = [d for d in grp["game_date"].tolist() if isinstance(d, date)]
        rests = [(start_dates[i] - start_dates[i - 1]).days for i in range(1, len(start_dates))]
        med_rest = float(pd.Series(rests).median()) if rests else math.nan
        wk_mode = grp["game_date"].map(lambda d: d.weekday()).mode()
        dom_wk = int(wk_mode.iloc[0]) if not wk_mode.empty else -1

        score = abs(days_since - target_rest)
        if not math.isnan(med_rest):
            score = min(score, abs(days_since - med_rest) + 0.25)
        if dom_wk == game_day.weekday():
            score -= 1.0
        score -= min(starts_n, 6) * 0.15
        if days_since < 3:
            score += 4.0
        if days_since > 14:
            score += 2.0

        conf = 0.35 + min(starts_n, 5) * 0.08
        if abs(days_since - target_rest) <= 1:
            conf += 0.12
        if dom_wk == game_day.weekday():
            conf += 0.12
        if days_since < 3 or days_since > 14:
            conf -= 0.20
        conf = max(0.10, min(0.85, conf))

        rows.append({
            "pitcher_espn_id": str(pid).strip(),
            "pitcher_name": str(last_row["pitcher_name"]).strip(),
            "score": float(score),
            "confidence": float(conf),
            "starts_n": starts_n,
            "days_since": days_since,
            "target_rest": target_rest,
        })
    if not rows:
        return None
    best = sorted(rows, key=lambda r: (r["score"], -r["starts_n"], r["days_since"]))[0]
    note = (
        f"inferred from ESPN starts: rest={best['days_since']}d, target={best['target_rest']}d, "
        f"team_starts={best['starts_n']}"
    )
    return StarterPick(
        espn_id=best["pitcher_espn_id"],
        name=best["pitcher_name"],
        source="inferred_rotation",
        confidence=best["confidence"],
        note=note,
    )


def load_index_map(path: Path, key_col: str, val_col: str) -> dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    if key_col not in df.columns or val_col not in df.columns:
        return {}
    out: dict[str, str] = {}
    for _, r in df.iterrows():
        k = str(r[key_col]).strip()
        v = str(r[val_col]).strip()
        if k:
            out[k] = v
    return out


def d1_slug_candidates(team_name: str) -> list[str]:
    n = normalize_team_name(team_name)
    tokens = n.split()
    if not tokens:
        return []
    candidates = set()

    def add_from_tokens(toks: list[str]) -> None:
        toks = [t for t in toks if t]
        if toks:
            candidates.add("-".join(toks))

    add_from_tokens(tokens)

    repl_sets = [
        ("st", "state"),
        ("state", "st"),
        ("saint", "st"),
        ("and", ""),
    ]
    for old, new in repl_sets:
        t2 = [new if t == old else t for t in tokens]
        add_from_tokens(t2)

    # Remove leading "the" and "university" variants.
    if tokens and tokens[0] == "the":
        add_from_tokens(tokens[1:])
    add_from_tokens([t for t in tokens if t != "university"])

    # ESPN sometimes has "n c state"; try condensed two-token form.
    if len(tokens) >= 3 and tokens[0] == "n" and tokens[1] == "c":
        add_from_tokens(["nc"] + tokens[2:])
    if len(tokens) >= 2 and tokens[0] == "u" and len(tokens[1]) == 1:
        add_from_tokens(["u" + tokens[1]] + tokens[2:])

    return sorted(candidates, key=lambda s: (len(s), s))


def fetch_d1_rows_for_slug(
    session: requests.Session,
    slug: str,
    year: int,
    rows_cache: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if slug in rows_cache:
        return rows_cache[slug]
    try:
        url = f"https://d1baseball.com/team/{slug}/lineup/"
        resp = session.get(url, timeout=20, headers={"User-Agent": UA})
        if resp.status_code != 200:
            rows_cache[slug] = []
            return rows_cache[slug]
        rows_cache[slug] = parse_d1_lineup_rows(resp.text, year=year)
        return rows_cache[slug]
    except Exception:
        rows_cache[slug] = []
        return rows_cache[slug]


def resolve_d1_slug_for_team(
    session: requests.Session,
    team_name: str,
    slug_cache: dict[str, str],
    rows_cache: dict[str, list[dict[str, Any]]],
    year: int,
) -> str:
    key = normalize_team_name(team_name)
    if not key:
        return ""
    if key in slug_cache:
        return slug_cache[key]
    for slug in d1_slug_candidates(team_name):
        rows = fetch_d1_rows_for_slug(session, slug, year, rows_cache)
        # Any parsed lineup rows means we found the correct team page.
        if rows:
            slug_cache[key] = slug
            return slug
    slug_cache[key] = ""
    return ""


def parse_d1_lineup_rows(html_text: str, year: int) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html_text, "html.parser")
    tables = soup.find_all("table")
    target_table = None
    for t in tables:
        headers = [h.get_text(" ", strip=True) for h in t.find_all("th")]
        if "SP" in headers and "Game" in headers:
            target_table = t
            break
    if target_table is None:
        return []
    headers = [h.get_text(" ", strip=True) for h in target_table.find_all("th")]
    try:
        sp_idx = headers.index("SP")
    except ValueError:
        sp_idx = len(headers) - 1

    rows: list[dict[str, Any]] = []
    for tr in target_table.find_all("tr")[1:]:
        tds = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        if not tds:
            continue
        game_text = tds[0]
        if "Most Games" in game_text or game_text.endswith("Games"):
            continue
        sp_name = tds[sp_idx] if sp_idx < len(tds) else ""
        dm = _GAME_DATE_RE.search(game_text)
        gdate: date | None = None
        if dm:
            dow_txt, mon_txt, day_txt = dm.groups()
            _ = dow_txt  # not used, retained for clarity
            try:
                gdate = datetime.strptime(f"{mon_txt} {day_txt} {year}", "%b %d %Y").date()
            except ValueError:
                gdate = None
        opp = ""
        om = _OPP_RE.search(game_text)
        if om:
            opp = om.group(1).strip()
        rows.append({
            "game_text": game_text,
            "game_date": gdate,
            "opponent": opp,
            "opponent_norm": normalize_team_name(opp),
            "sp_name": sp_name.strip(),
        })
    return rows


def choose_d1_starter(
    rows: list[dict[str, Any]],
    target_date: date,
    opponent_name: str,
) -> StarterPick | None:
    if not rows:
        return None
    candidates = [r for r in rows if r.get("game_date") == target_date]
    if not candidates:
        return None
    opp_norm = normalize_team_name(opponent_name)
    if opp_norm:
        matched = [r for r in candidates if r.get("opponent_norm") == opp_norm]
        if matched:
            candidates = matched
    c = candidates[0]
    sp_name = str(c.get("sp_name") or "").strip()
    if not sp_name or normalize_text(sp_name) in {"tbd", "to be determined"}:
        return StarterPick(
            espn_id="",
            name="",
            source="d1_lineup_today",
            confidence=0.20,
            note=f"D1 lineup row found but SP is TBD/blank ({c.get('game_text', '')})",
        )
    return StarterPick(
        espn_id="",
        name=sp_name,
        source="d1_lineup_today",
        confidence=0.78,
        note=f"D1 lineup row: {c.get('game_text', '')}",
    )


def write_manual_template(path: Path, games: list[dict[str, Any]], overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        print(f"Manual template exists; keeping as-is: {path}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for g in games:
        rows.append({
            "game_date": g["game_date"],
            "event_id": g["event_id"],
            "commence_time": g["commence_time"],
            "home_team": g["home_team_name"],
            "away_team": g["away_team_name"],
            "home_canonical_id": g["home_canonical_id"],
            "away_canonical_id": g["away_canonical_id"],
            "home_pitcher_name": "",
            "away_pitcher_name": "",
            "home_pitcher_espn_id": "",
            "away_pitcher_espn_id": "",
            "source": "",
            "source_url": "",
            "notes": "",
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Wrote manual template ({len(rows)} games): {path}")
    return True


def build_game_list(
    session: requests.Session,
    game_day: date,
    canonical: pd.DataFrame,
    name_to_canonical: dict[str, tuple[str, int]],
    canonical_name_index: dict[str, list[tuple[str, int, str]]],
    include_final: bool,
) -> list[dict[str, Any]]:
    scoreboard_url = f"{ESPN_BASE}/scoreboard?dates={game_day.strftime('%Y%m%d')}&limit=200"
    payload = fetch_json(session, scoreboard_url)
    if not payload:
        return []
    games: list[dict[str, Any]] = []
    for event in payload.get("events", []):
        comp = (event.get("competitions") or [{}])[0]
        status = ((comp.get("status") or {}).get("type") or {}).get("name", "")
        if not include_final and status in FINAL_STATUSES:
            continue
        competitors = comp.get("competitors") or []
        if len(competitors) < 2:
            continue
        home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
        away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
        home_name = str((home.get("team") or {}).get("displayName") or "").strip()
        away_name = str((away.get("team") or {}).get("displayName") or "").strip()
        if not home_name or not away_name:
            continue
        home_t = resolve_team_name_to_canonical(home_name, canonical, name_to_canonical, canonical_name_index)
        away_t = resolve_team_name_to_canonical(away_name, canonical, name_to_canonical, canonical_name_index)
        home_cid = home_t[0] if home_t else ""
        away_cid = away_t[0] if away_t else ""
        games.append({
            "event_id": str(event.get("id") or "").strip(),
            "game_date": game_day.isoformat(),
            "commence_time": str(event.get("date") or "").strip(),
            "status": status,
            "home_team_name": home_name,
            "away_team_name": away_name,
            "home_team_espn_id": str((home.get("team") or {}).get("id") or "").strip(),
            "away_team_espn_id": str((away.get("team") or {}).get("id") or "").strip(),
            "home_canonical_id": str(home_cid),
            "away_canonical_id": str(away_cid),
        })
    return games


def make_pick_from_manual_side(
    row: pd.Series,
    side: str,
    team_key: str,
    starts: pd.DataFrame,
) -> StarterPick | None:
    pid = str(row.get(f"{side}_pitcher_espn_id", "") or "").strip()
    pname = str(row.get(f"{side}_pitcher_name", "") or "").strip()
    src = str(row.get("source", "") or "").strip() or "manual"
    src_url = str(row.get("source_url", "") or "").strip()
    note = str(row.get("notes", "") or "").strip()
    if pid:
        base_note = "manual starter id"
        if src_url:
            base_note += f" ({src_url})"
        if note:
            base_note += f"; {note}"
        return StarterPick(
            espn_id=pid,
            name=pname,
            source="manual",
            confidence=0.98,
            note=base_note,
        )
    if pname:
        rid, rname = resolve_name_to_pitcher_id(team_key, pname, starts)
        if rid:
            base_note = f"manual name matched to ESPN id via history ({src})"
            if src_url:
                base_note += f" {src_url}"
            if note:
                base_note += f"; {note}"
            return StarterPick(
                espn_id=rid,
                name=rname or pname,
                source="manual_name_matched",
                confidence=0.92,
                note=base_note,
            )
        base_note = f"manual name provided but ESPN id unresolved ({src})"
        if src_url:
            base_note += f" {src_url}"
        if note:
            base_note += f"; {note}"
        return StarterPick(
            espn_id="",
            name=pname,
            source="manual_unresolved_name",
            confidence=0.45,
            note=base_note,
        )
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Build today's probable/announced NCAA D1 starters table.")
    parser.add_argument("--date", default=date.today().isoformat(), help="Game date YYYY-MM-DD (default: today)")
    parser.add_argument("--canonical", type=Path, default=Path("data/registries/canonical_teams_2026.csv"))
    parser.add_argument("--pitching-lines", type=Path, default=Path("data/processed/pitching_lines_espn.csv"))
    parser.add_argument("--run-event-pitcher-index", type=Path, default=Path("data/processed/run_event_pitcher_index.csv"))
    parser.add_argument("--run-event-team-index", type=Path, default=Path("data/processed/run_event_team_index.csv"))
    parser.add_argument("--manual-input", type=Path, default=None, help="Manual starter CSV path (optional)")
    parser.add_argument("--write-manual-template", action="store_true", help="Write a template CSV for manual starters")
    parser.add_argument("--manual-template-only", action="store_true", help="Write manual template and exit")
    parser.add_argument("--overwrite-manual-template", action="store_true", help="Overwrite existing manual template")
    parser.add_argument("--include-final", action="store_true", help="Include final games in output")
    parser.add_argument("--disable-d1-lineups", action="store_true", help="Disable D1Baseball same-day lineup scrape")
    parser.add_argument("--disable-espn-summary", action="store_true", help="Disable ESPN summary starter check")
    parser.add_argument("--out", type=Path, default=Path("data/processed/todays_starters.csv"))
    args = parser.parse_args()

    try:
        game_day = parse_date_str(args.date)
    except ValueError:
        print(f"Invalid --date {args.date!r}; expected YYYY-MM-DD")
        return 1

    manual_path = args.manual_input
    if manual_path is None:
        manual_path = Path(f"data/raw/starters_manual/manual_starters_{game_day.isoformat()}.csv")

    session = requests.Session()
    session.headers.update({"User-Agent": UA})

    canonical = load_canonical_teams(args.canonical)
    name_to_canonical = build_odds_name_to_canonical(canonical)
    canonical_name_index = build_canonical_name_index(canonical)
    canonical_name_by_id: dict[str, str] = {}
    for _, r in canonical.iterrows():
        cid = str(r.get("canonical_id") or "").strip()
        nm = str(r.get("team_name") or "").strip()
        if cid and nm and cid not in canonical_name_by_id:
            canonical_name_by_id[cid] = nm

    games = build_game_list(
        session=session,
        game_day=game_day,
        canonical=canonical,
        name_to_canonical=name_to_canonical,
        canonical_name_index=canonical_name_index,
        include_final=args.include_final,
    )
    if not games:
        print(f"No ESPN games found for {game_day.isoformat()} (after status filter).")
        return 1
    print(f"Found {len(games)} games on ESPN scoreboard for {game_day.isoformat()}")

    if args.write_manual_template:
        write_manual_template(manual_path, games, overwrite=args.overwrite_manual_template)
        if args.manual_template_only:
            return 0

    manual_df = load_manual_starters(manual_path)
    starts = load_pitching_starts(args.pitching_lines)
    pitcher_idx_map = load_index_map(args.run_event_pitcher_index, "pitcher_espn_id", "pitcher_idx")
    team_idx_map = load_index_map(args.run_event_team_index, "canonical_id", "team_idx")

    use_d1 = not args.disable_d1_lineups
    d1_slug_cache: dict[str, str] = {}
    d1_rows_cache: dict[str, list[dict[str, Any]]] = {}
    if use_d1:
        print("D1 same-day lineup probing enabled (slug candidates from ESPN team names)")

    summary_cache: dict[str, dict[str, dict[str, str]]] = {}
    use_summary = not args.disable_espn_summary

    out_rows: list[dict[str, Any]] = []
    for g in games:
        event_id = g["event_id"]
        home_cid = g["home_canonical_id"]
        away_cid = g["away_canonical_id"]
        home_team_key = home_cid or normalize_team_name(g["home_team_name"])
        away_team_key = away_cid or normalize_team_name(g["away_team_name"])

        home_pick: StarterPick | None = None
        away_pick: StarterPick | None = None

        # 1) Manual source
        man = find_manual_row(g, manual_df)
        if man is not None:
            home_pick = make_pick_from_manual_side(man, "home", home_team_key, starts)
            away_pick = make_pick_from_manual_side(man, "away", away_team_key, starts)

        # 2) D1 same-day lineup row
        if use_d1:
            for side in ("home", "away"):
                if side == "home" and home_pick is not None:
                    continue
                if side == "away" and away_pick is not None:
                    continue

                team_cid = home_cid if side == "home" else away_cid
                team_name = canonical_name_by_id.get(team_cid, g[f"{side}_team_name"])
                opp_name = g["away_team_name"] if side == "home" else g["home_team_name"]
                slug = resolve_d1_slug_for_team(
                    session=session,
                    team_name=team_name,
                    slug_cache=d1_slug_cache,
                    rows_cache=d1_rows_cache,
                    year=game_day.year,
                )
                if not slug:
                    continue
                rows = fetch_d1_rows_for_slug(session, slug, game_day.year, d1_rows_cache)
                pick = choose_d1_starter(rows, game_day, opp_name)
                if pick is None:
                    continue
                if pick.name and not pick.espn_id:
                    rid, rname = resolve_name_to_pitcher_id(
                        home_team_key if side == "home" else away_team_key,
                        pick.name,
                        starts,
                    )
                    if rid:
                        pick = StarterPick(
                            espn_id=rid,
                            name=rname or pick.name,
                            source=pick.source,
                            confidence=min(0.90, pick.confidence + 0.08),
                            note=pick.note + " | matched to ESPN id from history",
                        )
                    else:
                        pick = StarterPick(
                            espn_id="",
                            name=pick.name,
                            source=pick.source,
                            confidence=pick.confidence - 0.15,
                            note=pick.note + " | name unresolved to ESPN id",
                        )
                if side == "home":
                    home_pick = pick
                else:
                    away_pick = pick

        # 3) ESPN summary starter (in-progress / started)
        if use_summary and g["status"] not in SCHEDULED_STATUSES:
            if event_id not in summary_cache:
                summary = fetch_json(session, f"{ESPN_BASE}/summary?event={event_id}", retries=2)
                if summary:
                    summary_cache[event_id] = extract_starters_from_summary(
                        summary, g["home_team_espn_id"], g["away_team_espn_id"]
                    )
                else:
                    summary_cache[event_id] = {"home": {"id": "", "name": ""}, "away": {"id": "", "name": ""}}
            sm = summary_cache[event_id]
            if home_pick is None:
                hid = str((sm.get("home") or {}).get("id") or "").strip()
                hnm = str((sm.get("home") or {}).get("name") or "").strip()
                if hid or hnm:
                    home_pick = StarterPick(
                        espn_id=hid,
                        name=hnm,
                        source="espn_summary",
                        confidence=0.99 if hid else 0.80,
                        note="starter from ESPN summary boxscore",
                    )
            if away_pick is None:
                aid = str((sm.get("away") or {}).get("id") or "").strip()
                anm = str((sm.get("away") or {}).get("name") or "").strip()
                if aid or anm:
                    away_pick = StarterPick(
                        espn_id=aid,
                        name=anm,
                        source="espn_summary",
                        confidence=0.99 if aid else 0.80,
                        note="starter from ESPN summary boxscore",
                    )

        # 4) Inference fallback
        if home_pick is None:
            home_pick = infer_probable_starter(home_team_key, game_day, starts)
        if away_pick is None:
            away_pick = infer_probable_starter(away_team_key, game_day, starts)

        # 5) Unknown fallback
        if home_pick is None:
            home_pick = StarterPick("", "", "unknown", 0.0, "no source available")
        if away_pick is None:
            away_pick = StarterPick("", "", "unknown", 0.0, "no source available")

        # Best-effort name->ID resolution when a name exists but id is still blank.
        if not home_pick.espn_id and home_pick.name:
            rid, rname = resolve_name_to_pitcher_id(home_team_key, home_pick.name, starts)
            if rid:
                home_pick = StarterPick(
                    espn_id=rid,
                    name=rname or home_pick.name,
                    source=home_pick.source,
                    confidence=min(0.95, home_pick.confidence + 0.05),
                    note=home_pick.note + " | post-resolved ESPN id from history",
                )
        if not away_pick.espn_id and away_pick.name:
            rid, rname = resolve_name_to_pitcher_id(away_team_key, away_pick.name, starts)
            if rid:
                away_pick = StarterPick(
                    espn_id=rid,
                    name=rname or away_pick.name,
                    source=away_pick.source,
                    confidence=min(0.95, away_pick.confidence + 0.05),
                    note=away_pick.note + " | post-resolved ESPN id from history",
                )

        home_pid = home_pick.espn_id.strip()
        away_pid = away_pick.espn_id.strip()
        home_pidx = pitcher_idx_map.get(home_pid, "")
        away_pidx = pitcher_idx_map.get(away_pid, "")
        if not home_pidx:
            home_pidx = pitcher_idx_map.get("unknown", "0" if pitcher_idx_map else "")
        if not away_pidx:
            away_pidx = pitcher_idx_map.get("unknown", "0" if pitcher_idx_map else "")

        out_rows.append({
            "game_date": g["game_date"],
            "event_id": event_id,
            "commence_time": g["commence_time"],
            "status": g["status"],
            "home_team_name": g["home_team_name"],
            "away_team_name": g["away_team_name"],
            "home_team_espn_id": g["home_team_espn_id"],
            "away_team_espn_id": g["away_team_espn_id"],
            "home_canonical_id": home_cid,
            "away_canonical_id": away_cid,
            "home_team_idx": team_idx_map.get(home_cid, ""),
            "away_team_idx": team_idx_map.get(away_cid, ""),
            "home_pitcher_name": home_pick.name,
            "away_pitcher_name": away_pick.name,
            "home_pitcher_espn_id": home_pid,
            "away_pitcher_espn_id": away_pid,
            "home_pitcher_idx": home_pidx,
            "away_pitcher_idx": away_pidx,
            "home_starter_source": home_pick.source,
            "away_starter_source": away_pick.source,
            "home_starter_confidence": round(home_pick.confidence, 3),
            "away_starter_confidence": round(away_pick.confidence, 3),
            "home_starter_note": home_pick.note,
            "away_starter_note": away_pick.note,
            "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        })

    out_df = pd.DataFrame(out_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} rows -> {args.out}")

    for side in ("home", "away"):
        src_col = f"{side}_starter_source"
        src_counts = out_df[src_col].value_counts(dropna=False).to_dict()
        if src_counts:
            print(f"{side.title()} starter sources: {src_counts}")
    unresolved_ids = (
        (out_df["home_pitcher_espn_id"].astype(str).str.strip() == "")
        | (out_df["away_pitcher_espn_id"].astype(str).str.strip() == "")
    ).sum()
    print(f"Games with at least one missing pitcher ESPN id: {int(unresolved_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
