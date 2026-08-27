from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def install_state_dict_hooks(route: Any) -> None:
    if route._inputs and isinstance(route.target, torch.nn.Module) and "_inputs" in route.target._modules:
        raise ValueError("a routed target with nested inputs cannot have an '_inputs' child module")

    route._load_state_dict_prefix = None
    route._load_state_dict_errors = None
    route.register_state_dict_post_hook(_flatten_state_dict)
    route.register_load_state_dict_pre_hook(_expand_state_dict)
    route.register_load_state_dict_post_hook(_flatten_incompatible_keys)


def _rename_keys(state_dict: dict[str, Any], rename: Callable[[str], str]) -> None:
    items: list[tuple[str, Any]] = []
    keys: set[str] = set()

    for key, value in state_dict.items():
        renamed = rename(key)
        if renamed in keys:
            raise RuntimeError(f"state_dict key collision after route flattening: {renamed!r}")
        keys.add(renamed)
        items.append((renamed, value))

    state_dict.clear()
    state_dict.update(items)


def _flatten_state_dict(
    route: Any,
    state_dict: dict[str, Any],
    prefix: str,
    local_metadata: dict[str, Any],
) -> None:
    del local_metadata
    target_prefix = f"{prefix}target."
    inputs_prefix = f"{prefix}_inputs."

    def flatten(key: str) -> str:
        if not key.startswith(target_prefix):
            return key

        flattened = prefix + key.removeprefix(target_prefix)
        if route._inputs and flattened.startswith(inputs_prefix):
            raise RuntimeError("routed target state collides with the reserved '_inputs' namespace")
        return flattened

    _rename_keys(state_dict, flatten)


def _expand_state_dict(
    route: Any,
    state_dict: dict[str, Any],
    prefix: str,
    local_metadata: dict[str, Any],
    strict: bool,
    missing_keys: list[str],
    unexpected_keys: list[str],
    error_messages: list[str],
) -> None:
    del local_metadata, strict, missing_keys, unexpected_keys
    if not isinstance(route.target, torch.nn.Module):
        return

    route._load_state_dict_prefix = prefix
    route._load_state_dict_errors = error_messages
    target_prefix = f"{prefix}target."
    inputs_prefix = f"{prefix}_inputs."

    def expand(key: str) -> str:
        if not key.startswith(prefix):
            return key
        if route._inputs and key.startswith(inputs_prefix):
            return key
        return target_prefix + key.removeprefix(prefix)

    _rename_keys(state_dict, expand)


def _flatten_incompatible_keys(route: Any, incompatible_keys: Any) -> None:
    prefix = route._load_state_dict_prefix
    if prefix is None:
        return

    target_prefix = f"{prefix}target."
    for keys in (incompatible_keys.missing_keys, incompatible_keys.unexpected_keys):
        for index, key in enumerate(keys):
            if key.startswith(target_prefix):
                keys[index] = prefix + key.removeprefix(target_prefix)

    if route._load_state_dict_errors is not None:
        for index, message in enumerate(route._load_state_dict_errors):
            route._load_state_dict_errors[index] = message.replace(target_prefix, prefix)

    route._load_state_dict_prefix = None
    route._load_state_dict_errors = None
