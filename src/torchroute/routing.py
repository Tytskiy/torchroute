from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch

from ._state_dict import install_state_dict_hooks


def _add_exception_note(error: Exception, note: str) -> None:
    add_note = cast(Callable[[str], None] | None, getattr(error, "add_note", None))
    if add_note is not None:
        add_note(note)
    else:
        error.args = (*error.args, note)


class Ref(ABC):
    @abstractmethod
    def resolve(self, *, prev: Any, batch: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class _PathRef(Ref):
    root: str
    path: tuple[tuple[Literal["item", "attr"], Any], ...] = ()

    def __getitem__(self, key: Any) -> _PathRef:
        return type(self)(self.root, (*self.path, ("item", key)))

    def __getattr__(self, name: str) -> _PathRef:
        if name.startswith("__"):
            raise AttributeError(name)
        return type(self)(self.root, (*self.path, ("attr", name)))

    def resolve(self, *, prev: Any, batch: Any) -> Any:
        result = prev if self.root == "prev" else batch
        for kind, value in self.path:
            access = f"[{value!r}]" if kind == "item" else f".{value}"
            try:
                result = result[value] if kind == "item" else getattr(result, value)
            except Exception as error:
                _add_exception_note(error, f"while resolving {self} at {access}")
                raise
        return result

    def __repr__(self) -> str:
        path = "".join(f"[{value!r}]" if kind == "item" else f".{value}" for kind, value in self.path)
        return self.root + path


@dataclass(frozen=True, slots=True)
class _ValueRef(Ref):
    item: Any

    def resolve(self, *, prev: Any, batch: Any) -> Any:
        return self.item

    def __repr__(self) -> str:
        return f"value({self.item!r})"


prev = _PathRef("prev")
batch = _PathRef("batch")


def value(item: Any) -> Ref:
    return _ValueRef(item)


def _route_method(self: torch.nn.Module, *args: Any, **kwargs: Any) -> Route:
    return route(self, *args, **kwargs)


class Module(torch.nn.Module):
    """An ``nn.Module`` with stable ``.route(...)`` syntax."""

    route = _route_method


class _Node(Module):
    """An internal module evaluated with both routing-context values."""


def _map_structure(item: Any, map_leaf: Callable[[Any], Any]) -> Any:
    if isinstance(item, tuple):
        return tuple(_map_structure(value, map_leaf) for value in item)
    if isinstance(item, list):
        return [_map_structure(value, map_leaf) for value in item]
    if isinstance(item, dict):
        return {key: _map_structure(value, map_leaf) for key, value in item.items()}
    return map_leaf(item)


class Route(_Node):
    """An owning module call whose arguments are resolved at execution time."""

    def __init__(self, target: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        if not callable(target):
            raise TypeError(f"route target must be callable, got {type(target).__name__}")

        self.target: Callable[..., Any] = target
        self._inputs = torch.nn.ModuleList()
        self._args = cast(tuple[Any, ...], _map_structure(args, self._register_input))
        self._kwargs = cast(dict[str, Any], _map_structure(kwargs, self._register_input))
        install_state_dict_hooks(self)

    @property
    def args(self) -> tuple[Any, ...]:
        return self._args

    @property
    def kwargs(self) -> Mapping[str, Any]:
        return dict(self._kwargs)

    def _register_input(self, argument: Any) -> Any:
        if isinstance(argument, _Node):
            self._inputs.append(argument)
        return argument

    @staticmethod
    def _resolve(argument: Any, *, prev_value: Any, batch_value: Any) -> Any:
        if isinstance(argument, Ref):
            return argument.resolve(prev=prev_value, batch=batch_value)
        if isinstance(argument, _Node):
            return argument(prev=prev_value, batch=batch_value)
        return argument

    def forward(self, prev: Any = None, batch: Any = None) -> Any:
        def resolve(argument: Any) -> Any:
            return self._resolve(argument, prev_value=prev, batch_value=batch)

        args = _map_structure(self._args, resolve)
        kwargs = _map_structure(self._kwargs, resolve)
        return self.target(*args, **kwargs)

    def __repr__(self) -> str:
        arguments = [repr(argument) for argument in self.args]
        arguments.extend(f"{name}={argument!r}" for name, argument in self.kwargs.items())
        separator = ", " if arguments else ""
        return f"route({self.target!r}{separator}{', '.join(arguments)})"


def route(target: Callable[..., Any], *args: Any, **kwargs: Any) -> Route:
    """Create a routed call around a module or callable."""

    return Route(target, *args, **kwargs)


Step = _Node | Ref | torch.nn.Module | Callable[..., Any]


def _as_node(step: Step) -> _Node:
    if isinstance(step, _Node):
        return step
    if isinstance(step, Ref):
        return Route(lambda x: x, step)
    if callable(step):
        return Route(step, prev)
    raise TypeError(f"route step must be callable or a reference, got {type(step).__name__}")


def _indexed_steps(steps: tuple[Step, ...]) -> Iterator[tuple[str, Step]]:
    return ((str(index), step) for index, step in enumerate(steps))


class _Container(_Node):
    def __init__(self, entries: Iterable[tuple[str, Step]]) -> None:
        super().__init__()
        for name, step in entries:
            self.add_module(name, _as_node(step))

        if not self._modules:
            raise ValueError(f"{type(self).__name__} requires at least one step")

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[_Node]:
        return (cast(_Node, module) for module in self._modules.values())

    def __getitem__(self, name: str | int) -> _Node:
        if isinstance(name, int):
            return tuple(self)[name]
        return cast(_Node, self._modules[name])


class _Sequence(_Container):
    def __init__(self, *steps: Step) -> None:
        super().__init__(_indexed_steps(steps))

    def _run(self, prev: Any, batch: Any) -> Any:
        for step in self:
            prev = step(prev=prev, batch=batch)
        return prev


class Sequential(_Sequence):
    def forward(self, prev: Any = None, batch: Any = None) -> Any:
        return self._run(prev, batch)


class Parallel(_Container):
    def __init__(self, *steps: Step) -> None:
        super().__init__(_indexed_steps(steps))

    def forward(self, prev: Any = None, batch: Any = None) -> tuple[Any, ...]:
        return tuple(step(prev=prev, batch=batch) for step in self)


class NamedParallel(_Container):
    def __init__(self, **steps: Step) -> None:
        super().__init__(steps.items())

    def forward(self, prev: Any = None, batch: Any = None) -> dict[str, Any]:
        return {name: cast(_Node, step)(prev=prev, batch=batch) for name, step in self._modules.items()}


class Sum(_Container):
    def __init__(self, *steps: Step) -> None:
        super().__init__(_indexed_steps(steps))

    def forward(self, prev: Any = None, batch: Any = None) -> Any:
        iterator = iter(self)
        result = next(iterator)(prev=prev, batch=batch)
        for step in iterator:
            result = result + step(prev=prev, batch=batch)
        return result


class Concat(_Container):
    def __init__(self, *steps: Step, dim: int = -1) -> None:
        super().__init__(_indexed_steps(steps))
        self.dim = dim

    def forward(self, prev: Any = None, batch: Any = None) -> torch.Tensor:
        outputs = [step(prev=prev, batch=batch) for step in self]
        return torch.cat(outputs, dim=self.dim)


class Model(_Sequence):
    """A batch-only root around a routed computation."""

    def forward(self, batch: Any) -> Any:
        return self._run(prev=None, batch=batch)


__all__ = [
    "Concat",
    "Model",
    "Module",
    "NamedParallel",
    "Parallel",
    "Ref",
    "Route",
    "Sequential",
    "Sum",
    "batch",
    "prev",
    "route",
    "value",
]
