from __future__ import annotations

import inspect
from threading import RLock
from typing import Any

import torch

from .routing import _route_method

_MISSING = object()
_LOCK = RLock()


def enable_module_routes() -> None:
    """Add ``.route(...)`` syntax to every ``nn.Module``."""

    with _LOCK:
        existing: Any = inspect.getattr_static(torch.nn.Module, "route", _MISSING)
        if existing is _route_method:
            return
        if existing is not _MISSING:
            raise RuntimeError("torch.nn.Module.route already exists and was not installed by torchroute")
        torch.nn.Module.route = _route_method  # type: ignore[attr-defined]


def disable_module_routes() -> None:
    """Remove torchroute's ``.route(...)`` method from ``nn.Module``."""

    with _LOCK:
        existing: Any = inspect.getattr_static(torch.nn.Module, "route", _MISSING)
        if existing is _MISSING:
            return
        if existing is not _route_method:
            raise RuntimeError("torch.nn.Module.route was replaced by another implementation")
        del torch.nn.Module.route  # type: ignore[attr-defined]


def is_module_routes_enabled() -> bool:
    """Return whether global module route syntax is enabled."""

    return inspect.getattr_static(torch.nn.Module, "route", _MISSING) is _route_method


__all__ = [
    "disable_module_routes",
    "enable_module_routes",
    "is_module_routes_enabled",
]
