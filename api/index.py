"""
============================================================
SCALABLE ASSOCIATION RULE MINING
Vercel Serverless Entry Point
============================================================

Exposes the canonical Flask application as `app` for Vercel's
Python serverless runtime.
============================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root and webapp are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

WEBAPP_DIR = PROJECT_ROOT / "webapp"
if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))

from webapp.app import app

__all__ = ["app"]
