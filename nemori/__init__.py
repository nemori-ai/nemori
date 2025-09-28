"""Public package alias that re-exports the core Nemori API."""

from importlib import import_module

_src = import_module("src")

__all__ = getattr(_src, "__all__", [])

globals().update({name: getattr(_src, name) for name in __all__})
