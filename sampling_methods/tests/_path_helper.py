"""Utilities to make the tests runnable as standalone scripts."""

from __future__ import annotations

import os
import sys


def ensure_project_root_on_path() -> None:
    """Prepend the repository root to sys.path if it is missing."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
