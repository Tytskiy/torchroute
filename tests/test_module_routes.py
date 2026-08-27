from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch

import torchroute as tr
from torchroute.module_routes import (
    disable_module_routes,
    enable_module_routes,
    is_module_routes_enabled,
)


@pytest.fixture(autouse=True)
def module_routes_are_disabled() -> Iterator[None]:
    disable_module_routes()
    yield
    disable_module_routes()


def test_global_module_route_syntax_is_explicit_idempotent_and_reversible() -> None:
    assert tr.enable_module_routes is enable_module_routes
    assert tr.disable_module_routes is disable_module_routes
    assert tr.is_module_routes_enabled is is_module_routes_enabled
    assert not is_module_routes_enabled()
    assert not hasattr(torch.nn.Module, "route")

    enable_module_routes()
    enable_module_routes()

    assert is_module_routes_enabled()
    routed = torch.nn.Linear(2, 1).route(tr.batch["x"])  # type: ignore[operator]
    model = tr.Model(routed)

    disable_module_routes()
    disable_module_routes()

    assert not is_module_routes_enabled()
    assert not hasattr(torch.nn.Module, "route")
    assert model({"x": torch.ones(3, 2)}).shape == (3, 1)


def test_enable_refuses_to_overwrite_an_existing_method() -> None:
    def foreign_route(self: torch.nn.Module) -> None:
        return None

    torch.nn.Module.route = foreign_route  # type: ignore[attr-defined]
    try:
        with pytest.raises(RuntimeError, match="was not installed by torchroute"):
            enable_module_routes()
    finally:
        del torch.nn.Module.route  # type: ignore[attr-defined]
