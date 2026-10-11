"""Vercel entry point: serves the LogiSmart JSON API from server.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server import app  # noqa: E402,F401
