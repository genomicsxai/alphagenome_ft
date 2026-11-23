"""Detect Kaggle API credentials for tests that download AlphaGenome weights."""

from __future__ import annotations

import json
import os
from pathlib import Path


def kaggle_credentials_available() -> bool:
    """True if any credential form kagglehub accepts is present.

    Covers ``KAGGLE_USERNAME``/``KAGGLE_KEY``, ``~/.kaggle/kaggle.json``,
    and the newer single-token forms (``KAGGLE_API_TOKEN`` env var or
    ``~/.kaggle/access_token``) — delegates to kagglehub's own credential
    resolution so this stays correct as that library's supported forms
    evolve.
    """
    try:
        import kagglehub

        if kagglehub.config.get_kaggle_credentials() is not None:
            return True
    except ImportError:
        pass

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
