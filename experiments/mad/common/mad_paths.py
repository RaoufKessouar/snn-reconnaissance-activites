"""Centralised and portable locations for the MAD data.

The original scripts embedded one user's absolute server paths.  Keeping path
resolution here makes the same code usable on another account or machine.
Environment variables always take precedence over repository-local defaults.
"""

from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("SRESNET_DATA_ROOT", REPOSITORY_ROOT / "data")).expanduser()
MAD_ROOT = Path(os.environ.get("MAD_DATA_ROOT", DATA_ROOT / "lab_gesture_dataset")).expanduser()

ZIP_DIR = Path(os.environ.get("MAD_ZIP_DIR", MAD_ROOT / "zips")).expanduser()
EXTRACT_DIR = Path(os.environ.get("MAD_EXTRACT_DIR", MAD_ROOT / "extracted")).expanduser()
CACHE_DIR = Path(os.environ.get("MAD_CACHE_DIR", DATA_ROOT / "MAD_cache")).expanduser()
