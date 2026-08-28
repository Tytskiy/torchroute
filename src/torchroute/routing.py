from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast

import torch


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


def _map_structure(item: Any, map_leaf: Callable[[Any], Any]) -> Any:
    if isinstance(item, tuple):
        return tuple(_map_structure(value, map_leaf) for value in item)
    if isinstance(item, list):
        return [_map_structure(value, map_leaf) for value in item]
    if isinstance(item, dict):
        return {key: _map_structure(value, map_leaf) for key, value in item.items()}
    return map_leaf(item)


def _resolve_structure(item: Any, *, prev_value: Any, batch_value: Any) -> Any:
    def resolve(argument: Any) -> Any:
        if isinstance(argument, Ref):
            return argument.resolve(prev=prev_value, batch=batch_value)
        return argument

    return _map_structure(item, resolve)


class Route:
    """A call specification materialized by torchroute containers."""

    __slots__ = ("_args", "_kwargs", "_target")

    def __init__(self, target: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        if not callable(target):
            raise TypeError(f"route target must be callable, got {type(target).__name__}")

        self._target = target
        self._args = cast(tuple[Any, ...], _prepare_route_arguments(args))
        self._kwargs = cast(dict[str, Any], _prepare_route_arguments(kwargs))

    @property
    def target(self) -> Callable[..., Any]:
        return self._target

    @property
    def args(self) -> tuple[Any, ...]:
        return self._args

    @property
    def kwargs(self) -> Mapping[str, Any]:
        return dict(self._kwargs)

    def as_module(self) -> Sequential:
        """Materialize this route as a standalone module."""

        return Sequential(self)

    def __repr__(self) -> str:
        arguments = [repr(argument) for argument in self.args]
        arguments.extend(f"{name}={argument!r}" for name, argument in self.kwargs.items())
        separator = ", " if arguments else ""
        return f"route({self.target!r}{separator}{', '.join(arguments)})"


def _route_method(self: torch.nn.Module, *args: Any, **kwargs: Any) -> Route:
    return route(self, *args, **kwargs)


class Module(torch.nn.Module):
    """An ``nn.Module`` with stable ``.route(...)`` syntax."""

    route = _route_method


def _prepare_route_arguments(item: Any) -> Any:
    def validate(argument: Any) -> Any:
        if isinstance(argument, Route):
            raise TypeError("route arguments cannot contain another route; make it a separate container step")
        if isinstance(argument, torch.nn.Module):
            raise TypeError("route arguments cannot contain an nn.Module; make it a separate container step")
        return argument

    return _map_structure(item, validate)


def route(target: Callable[..., Any], *args: Any, **kwargs: Any) -> Route:
    return Route(target, *args, **kwargs)


class _Plan(Protocol):
    def run(self, owner: _Container, *, prev_value: Any, batch_value: Any) -> Any: ...

    def exposed(self, owner: _Container) -> Any: ...


def _invoke(
    target: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    *,
    prev_value: Any,
    batch_value: Any,
) -> Any:
    resolved_args = cast(
        tuple[Any, ...], _resolve_structure(args, prev_value=prev_value, batch_value=batch_value)
    )
    resolved_kwargs = cast(
        dict[str, Any], _resolve_structure(kwargs, prev_value=prev_value, batch_value=batch_value)
    )
    return target(*resolved_args, **resolved_kwargs)


@dataclass(frozen=True, slots=True)
class _ModulePlan:
    name: str
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def run(self, owner: _Container, *, prev_value: Any, batch_value: Any) -> Any:
        return _invoke(
            cast(Callable[..., Any], owner.get_submodule(self.name)),
            self.args,
            self.kwargs,
            prev_value=prev_value,
            batch_value=batch_value,
        )

    def exposed(self, owner: _Container) -> Any:
        return owner.get_submodule(self.name)


@dataclass(frozen=True, slots=True)
class _CallablePlan:
    target: Callable[..., Any]
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def run(self, owner: _Container, *, prev_value: Any, batch_value: Any) -> Any:
        return _invoke(
            self.target,
            self.args,
            self.kwargs,
            prev_value=prev_value,
            batch_value=batch_value,
        )

    def exposed(self, owner: _Container) -> Any:
        return self.target


@dataclass(frozen=True, slots=True)
class _RefPlan:
    ref: Ref

    def run(self, owner: _Container, *, prev_value: Any, batch_value: Any) -> Any:
        return self.ref.resolve(prev=prev_value, batch=batch_value)

    def exposed(self, owner: _Container) -> Any:
        return self.ref


Step = Route | Ref | torch.nn.Module | Callable[..., Any]


def _indexed_steps(steps: tuple[Step, ...]) -> Iterator[tuple[str, Step]]:
    return ((str(index), step) for index, step in enumerate(steps))


class _Container(Module):
    def __init__(self, entries: Iterable[tuple[str, Step]]) -> None:
        super().__init__()
        plans: list[tuple[str, _Plan]] = [(name, self._prepare_plan(name, step)) for name, step in entries]
        if not plans:
            raise ValueError(f"{type(self).__name__} requires at least one step")
        self._plans = tuple(plans)

    def _prepare_target(
        self,
        name: str,
        target: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> _Plan:
        if isinstance(target, torch.nn.Module):
            self.add_module(name, target)
            return _ModulePlan(name, args, kwargs)
        return _CallablePlan(target, args, kwargs)

    def _prepare_plan(self, name: str, step: Step) -> _Plan:
        if isinstance(step, Route):
            return self._prepare_target(name, step.target, step.args, step.kwargs)

        if isinstance(step, Ref):
            return _RefPlan(step)

        if isinstance(step, _Container):
            return self._prepare_target(name, step, (prev,), {"batch": batch})

        if isinstance(step, torch.nn.Module):
            return self._prepare_target(name, step, (prev,), {})

        if callable(step):
            return self._prepare_target(name, step, (prev,), {})

        raise TypeError(f"route step must be callable or a reference, got {type(step).__name__}")

    def __len__(self) -> int:
        return len(self._plans)

    def __iter__(self) -> Iterator[Step]:
        return (cast(Step, plan.exposed(self)) for _, plan in self._plans)

    def __getitem__(self, name: str | int) -> Step:
        if isinstance(name, int):
            return cast(Step, self._plans[name][1].exposed(self))
        for plan_name, plan in self._plans:
            if plan_name == name:
                return cast(Step, plan.exposed(self))
        raise KeyError(name)

    def _execute(self, plan: _Plan, *, prev_value: Any, batch_value: Any) -> Any:
        return plan.run(self, prev_value=prev_value, batch_value=batch_value)


class _Sequence(_Container):
    def __init__(self, *steps: Step) -> None:
        super().__init__(_indexed_steps(steps))

    def _run(self, prev_value: Any, batch_value: Any) -> Any:
        for _, plan in self._plans:
            prev_value = self._execute(plan, prev_value=prev_value, batch_value=batch_value)
        return prev_value


class Sequential(_Sequence):
    def forward(self, prev: Any = None, *, batch: Any = None) -> Any:
        return self._run(prev, batch)


class Parallel(_Container):
    def __init__(self, *steps: Step) -> None:
        super().__init__(_indexed_steps(steps))

    def forward(self, prev: Any = None, *, batch: Any = None) -> tuple[Any, ...]:
        return tuple(self._execute(plan, prev_value=prev, batch_value=batch) for _, plan in self._plans)


class NamedParallel(_Container):
    def __init__(self, **steps: Step) -> None:
        super().__init__(steps.items())

    def forward(self, prev: Any = None, *, batch: Any = None) -> dict[str, Any]:
        return {name: self._execute(plan, prev_value=prev, batch_value=batch) for name, plan in self._plans}


class Sum(_Container):
    def __init__(self, *steps: Step) -> None:
        super().__init__(_indexed_steps(steps))

    def forward(self, prev: Any = None, *, batch: Any = None) -> Any:
        iterator = iter(self._plans)
        _, first = next(iterator)
        result = self._execute(first, prev_value=prev, batch_value=batch)
        for _, plan in iterator:
            result = result + self._execute(plan, prev_value=prev, batch_value=batch)
        return result


class Concat(_Container):
    def __init__(self, *steps: Step, dim: int = -1) -> None:
        super().__init__(_indexed_steps(steps))
        self.dim = dim

    def forward(self, prev: Any = None, *, batch: Any = None) -> torch.Tensor:
        outputs = [self._execute(plan, prev_value=prev, batch_value=batch) for _, plan in self._plans]
        return torch.cat(outputs, dim=self.dim)


class Model(_Sequence):
    """A batch-only root around a routed computation."""

    def forward(self, batch: Any) -> Any:
        return self._run(prev_value=None, batch_value=batch)


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
