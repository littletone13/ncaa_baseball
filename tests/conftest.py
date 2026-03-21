from __future__ import annotations

import sys
from pathlib import Path


def _ensure_paths() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    scripts_path = repo_root / "scripts"
    for p in (src_path, scripts_path):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)


_ensure_paths()
