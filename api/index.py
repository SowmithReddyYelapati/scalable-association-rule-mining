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


class VercelPathFixMiddleware:
    """
    WSGI middleware that restores the original request path from
    Vercel routing headers (e.g. HTTP_X_MATCHED_PATH) so Flask
    can route sub-paths like /upload, /health, /learn, /analyze correctly.
    """
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        matched_path = (
            environ.get("HTTP_X_MATCHED_PATH")
            or environ.get("HTTP_X_NOW_ROUTE_MATCHES")
            or environ.get("HTTP_X_VERCEL_MATCHED_PATH")
            or environ.get("HTTP_X_FORWARDED_PATH")
        )
        if matched_path and matched_path not in ("/api/index", "/api/index.py"):
            path = matched_path.split("?")[0]
            if path:
                environ["PATH_INFO"] = path

        return self.wsgi_app(environ, start_response)


app.wsgi_app = VercelPathFixMiddleware(app.wsgi_app)

__all__ = ["app"]
