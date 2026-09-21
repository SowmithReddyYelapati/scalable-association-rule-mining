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

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from webapp.app import app

# Vercel discovers the WSGI application exported as `app`
__all__ = ["app"]
