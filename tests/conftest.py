"""Shared test fixtures and configuration."""
import sys
import types


def _ensure_lightweight_src():
    """Pre-populate src package to avoid heavy __init__.py imports."""
    if "src" not in sys.modules:
        src = types.ModuleType("src")
        src.__path__ = ["src"]
        sys.modules["src"] = src
    if "src.domain" not in sys.modules:
        domain = types.ModuleType("src.domain")
        domain.__path__ = ["src/domain"]
        sys.modules["src.domain"] = domain


_ensure_lightweight_src()
