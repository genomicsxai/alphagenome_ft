"""Detect Kaggle API credentials for tests that download AlphaGenome weights."""

from __future__ import annotations

import json
import os
from pathlib import Path


def kaggle_credentials_available() -> bool:
    """True if env vars or ``~/.kaggle/kaggle.json`` provide username + key."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.is_file():
        return False
    try:
        data = json.loads(kaggle_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(data.get("username") and data.get("key"))
