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
import traceback
from pathlib import Path

# Ensure repository root and webapp are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

WEBAPP_DIR = PROJECT_ROOT / "webapp"
if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))

try:
    from webapp.app import app as flask_app

    class WSGILoggingApp:
        """WSGI wrapper that intercepts exceptions and displays helpful debug info."""
        def __init__(self, inner_app):
            self.inner_app = inner_app

        def __call__(self, environ, start_response):
            try:
                return self.inner_app(environ, start_response)
            except Exception as exc:
                tb = traceback.format_exc()
                print("WSGI Exception caught:", tb, file=sys.stderr)
                status = "500 Internal Server Error"
                headers = [("Content-Type", "text/html; charset=utf-8")]
                start_response(status, headers)
                return [
                    f"<h2>ARM Application Error</h2><pre style='background:#f4f4f4;padding:12px;border:1px solid #ccc;'>{tb}</pre>".encode("utf-8")
                ]

    app = WSGILoggingApp(flask_app)
    handler = app

except Exception as init_exc:
    init_tb = traceback.format_exc()
    print("Flask Startup Exception:", init_tb, file=sys.stderr)

    def fallback_handler(environ, start_response):
        status = "500 Internal Server Error"
        headers = [("Content-Type", "text/html; charset=utf-8")]
        start_response(status, headers)
        return [
            f"<h2>ARM Startup Error</h2><pre style='background:#f4f4f4;padding:12px;border:1px solid #ccc;'>{init_tb}</pre>".encode("utf-8")
        ]

    app = fallback_handler
    handler = fallback_handler

__all__ = ["app", "handler"]
