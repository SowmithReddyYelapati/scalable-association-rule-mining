"""
============================================================
SCALABLE ASSOCIATION RULE MINING
Compatibility Entry Point
============================================================

The canonical Flask application is located at:
    webapp/app.py

This file serves as a convenience entry point that forwards
execution to the canonical webapp module.
============================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from webapp.app import app, BASE_DIR

if __name__ == "__main__":
    print()
    print("=" * 70)
    print("SCALABLE ASSOCIATION RULE MINING")
    print("=" * 70)
    print("Algorithms : Apriori + FP-Growth")
    print("UI         : Flask")
    print("Project    :", BASE_DIR)
    print("URL        : http://127.0.0.1:5000")
    print("=" * 70)
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )
