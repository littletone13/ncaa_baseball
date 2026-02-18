from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Team:
    id: str
    school: str
    mascot: str
    conference: str


def parse_teams_yaml(text: str) -> list[Team]:
    """
    Minimal, strict parser for `teams_baseball.yaml`-style files.

    We intentionally avoid PyYAML at bootstrap time. This parser is designed
    for the specific, simple structure of our checked-in YAML:

      teams:
        - id: "BSB_..."
          school: "..."
          mascot: "..."
          conference: "..."
    """
    teams: list[Team] = []
    cur: dict[str, str] = {}
    in_teams = False

    def strip_quotes(value: str) -> str:
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            return value[1:-1]
        return value

    def flush() -> None:
        nonlocal cur
        if not cur:
            return
        missing = [k for k in ("id", "school", "mascot", "conference") if k not in cur]
        if missing:
            raise ValueError(f"Team entry missing {missing}: {cur}")
        teams.append(
            Team(
                id=cur["id"],
                school=cur["school"],
                mascot=cur["mascot"],
                conference=cur["conference"],
            )
        )
        cur = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "teams:":
            in_teams = True
            continue
        if not in_teams:
            continue

        if line.startswith("- "):
            flush()
            line = line[2:].strip()

        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = strip_quotes(value.strip())
        cur[key] = value

    flush()

    ids = [t.id for t in teams]
    if len(ids) != len(set(ids)):
        seen: set[str] = set()
        dupes: list[str] = []
        for x in ids:
            if x in seen and x not in dupes:
                dupes.append(x)
            seen.add(x)
        raise ValueError(f"Duplicate team ids: {sorted(dupes)}")
    return teams

