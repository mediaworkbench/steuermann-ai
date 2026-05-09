from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import tomllib


@lru_cache(maxsize=1)
def get_framework_version() -> str:
    """Read framework version from pyproject.toml.

    Falls back to "unknown" when the file is missing or malformed.
    """
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    try:
        with pyproject_path.open("rb") as f:
            data = tomllib.load(f)
        return str(data.get("tool", {}).get("poetry", {}).get("version", "unknown"))
    except Exception:
        return "unknown"
