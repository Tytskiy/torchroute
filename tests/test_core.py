from __future__ import annotations

from typing import Any, cast

import pytest
import torch

import torchroute as tr


def test_refs_are_immutable_and_support_item_and_attribute_paths() -> None:
    user = tr.batch["user"]
    user_id = user["id"]
    user_age = user.age

    class User:
        age = 42
        id = 7

        def __getitem__(self, key: str) -> Any:
            return getattr(self, key)

    payload = {"user": User()}

    assert user_id.resolve(prev=None, batch=payload) == 7
    assert user_age.resolve(prev=None, batch=payload) == 42
    assert repr(user) == "batch['user']"
    assert repr(user_id) == "batch['user']['id']"


def test_module_route_and_route_function_have_the_same_semantics() -> None:
    class Add(tr.Module):
        def forward(self, left: int, right: int, scale: int = 1) -> int:
            return (left + right) * scale

    routed_method = Add().route(left=tr.batch["left"], right=tr.batch["right"], scale=2)
    routed_function = tr.route(Add(), left=tr.batch["left"], right=tr.batch["right"], scale=2)

    for routed in (routed_method, routed_function):
        model = tr.Model(routed)
        assert model({"left": 3, "right": 4}) == 14


def test_route_is_an_owning_module_and_can_be_executed_directly() -> None:
    linear = torch.nn.Linear(2, 1)
    routed = tr.route(linear, tr.prev)
    x = torch.ones(3, 2, requires_grad=True)

    result = routed(prev=x, batch={})
    result.sum().backward()

    assert result.shape == (3, 1)
    assert isinstance(routed, torch.nn.Module)
    assert routed.target is linear
    assert list(routed.state_dict()) == ["weight", "bias"]
    assert linear.weight.grad is not None


def test_model_is_a_batch_only_sequential_root() -> None:
    model = tr.Model(
        tr.route(lambda x: x + 1, tr.batch["x"]),
        lambda x: x * 2,
        lambda x: x - 3,
    )

    assert model({"x": 10}) == 19


def test_model_is_a_batch_only_sequence() -> None:
    model = tr.Model(tr.route(lambda x: x + 1, tr.batch["x"]), lambda x: x * 2)

    assert len(model) == 2
    assert isinstance(model[0], tr.Route)
    assert model({"x": 4}) == 10


def test_routes_can_own_nested_routes_in_structured_arguments() -> None:
    inner = tr.route(
        lambda left, right: left + right,
        tr.batch["left"],
        tr.batch["right"],
    )
    outer = tr.route(lambda values: values["result"], {"result": inner})
    model = tr.Model(outer)

    assert model({"left": 3, "right": 4}) == 7
    assert outer._inputs[0] is inner


def test_parallel_and_named_parallel_have_distinct_outputs() -> None:
    positional = tr.Model(
        tr.route(lambda x: x, tr.batch["x"]),
        tr.Parallel(lambda x: x, lambda x: x + 1),
    )
    named = tr.Model(
        tr.route(lambda x: x, tr.batch["x"]),
        tr.NamedParallel(original=lambda x: x, doubled=lambda x: x * 2),
    )

    x = torch.tensor([[1.0, 2.0]])
    pair = positional({"x": x})
    outputs = named({"x": x})

    assert pair[0] is x
    assert torch.equal(pair[1], x + 1)
    assert outputs["original"] is x
    assert torch.equal(outputs["doubled"], x * 2)


def test_sum_concat_and_reference_steps() -> None:
    model = tr.Model(
        tr.NamedParallel(
            summed=tr.Sum(tr.batch["x"], tr.route(lambda x: x * 2, tr.batch["x"])),
            concatenated=tr.Concat(tr.batch["x"], tr.route(lambda x: x + 1, tr.batch["x"])),
        )
    )

    x = torch.tensor([[1.0, 2.0]])
    result = model({"x": x})

    assert torch.equal(result["summed"], x * 3)
    assert torch.equal(result["concatenated"], torch.tensor([[1.0, 2.0, 2.0, 3.0]]))


def test_value_can_be_used_as_a_step() -> None:
    assert tr.Model(tr.value(42))({}) == 42


def test_custom_ref() -> None:
    class BatchSize(tr.Ref):
        def resolve(self, *, prev: Any, batch: Any) -> int:
            return len(batch["items"])

    model = tr.Model(tr.route(lambda size: size * 2, BatchSize()))
    assert model({"items": [1, 2, 3]}) == 6


def test_module_lifecycle_follows_the_natural_ownership_tree() -> None:
    linear = torch.nn.Linear(2, 1)
    model = tr.Model(tr.route(linear, tr.batch["x"]))

    assert list(model.state_dict()) == ["0.weight", "0.bias"]
    assert list(model.named_parameters()) == [
        ("0.target.weight", linear.weight),
        ("0.target.bias", linear.bias),
    ]

    model.eval()
    assert not model.training
    assert not model[0].training
    assert not linear.training

    model.to(dtype=torch.float64)
    assert linear.weight.dtype == torch.float64


def test_nested_modules_have_one_natural_ownership_path() -> None:
    inner = torch.nn.Linear(2, 2)
    outer = torch.nn.Linear(2, 1)
    model = tr.Model(tr.route(outer, tr.route(inner, tr.batch["x"])))

    assert list(model.state_dict()) == [
        "0.weight",
        "0.bias",
        "0._inputs.0.weight",
        "0._inputs.0.bias",
    ]
    assert model({"x": torch.ones(3, 2)}).shape == (3, 1)

    clone = tr.Model(
        tr.route(
            torch.nn.Linear(2, 1),
            tr.route(torch.nn.Linear(2, 2), tr.batch["x"]),
        )
    )
    assert not clone.load_state_dict(model.state_dict()).missing_keys
    assert torch.equal(
        model({"x": torch.ones(3, 2)}),
        clone({"x": torch.ones(3, 2)}),
    )


def test_shared_modules_follow_normal_pytorch_aliasing() -> None:
    shared = torch.nn.Linear(2, 2)
    model = tr.Model(
        tr.route(shared, tr.batch["x"]),
        tr.route(shared, tr.prev),
    )

    assert model[0].target is shared
    assert model[1].target is shared
    assert list(model.state_dict()) == ["0.weight", "0.bias", "1.weight", "1.bias"]
    assert model({"x": torch.ones(3, 2)}).shape == (3, 2)


def test_transparent_state_dict_preserves_buffers_and_loads_strictly() -> None:
    source = tr.Model(tr.route(torch.nn.BatchNorm1d(3), tr.batch["x"]))
    source({"x": torch.randn(8, 3)})

    state = source.state_dict()
    assert list(state) == [
        "0.weight",
        "0.bias",
        "0.running_mean",
        "0.running_var",
        "0.num_batches_tracked",
    ]

    target = tr.Model(tr.route(torch.nn.BatchNorm1d(3), tr.batch["x"]))
    incompatible = target.load_state_dict(state, strict=True)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    target_batch_norm = cast(torch.nn.BatchNorm1d, target[0].target)
    source_batch_norm = cast(torch.nn.BatchNorm1d, source[0].target)
    assert target_batch_norm.running_mean is not None
    assert source_batch_norm.running_mean is not None
    assert torch.equal(target_batch_norm.running_mean, source_batch_norm.running_mean)


def test_route_state_dict_is_transparent_inside_an_ordinary_module() -> None:
    class Wrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = tr.route(torch.nn.Linear(2, 1), tr.prev)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return cast(torch.Tensor, self.encoder(prev=value))

    source = Wrapper()
    state = source.state_dict()
    target = Wrapper()

    assert list(state) == ["encoder.weight", "encoder.bias"]
    assert not target.load_state_dict(state).missing_keys
    assert torch.equal(source(torch.ones(2, 2)), target(torch.ones(2, 2)))


def test_load_errors_use_transparent_state_names() -> None:
    model = tr.Model(tr.route(torch.nn.Linear(2, 2), tr.batch["x"]))
    missing_weight = model.state_dict()
    del missing_weight["0.weight"]

    incompatible = model.load_state_dict(missing_weight, strict=False)
    assert incompatible.missing_keys == ["0.weight"]

    wrong_shape = model.state_dict()
    wrong_shape["0.weight"] = torch.ones(3, 3)
    with pytest.raises(RuntimeError, match=r"size mismatch for 0\.weight") as error:
        model.load_state_dict(wrong_shape)
    assert "target" not in str(error.value)


def test_nested_inputs_reserve_the_inputs_state_namespace() -> None:
    class ConflictingTarget(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._inputs = torch.nn.Linear(2, 2)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return cast(torch.Tensor, self._inputs(value))

    nested = tr.route(torch.nn.Linear(2, 2), tr.batch["x"])
    with pytest.raises(ValueError, match="cannot have an '_inputs' child"):
        tr.route(ConflictingTarget(), nested)


def test_model_supports_torch_compile() -> None:
    model = tr.Model(tr.route(torch.nn.Linear(2, 1), tr.batch["x"]))
    compiled = torch.compile(model, backend="eager", fullgraph=True)

    assert compiled({"x": torch.ones(3, 2)}).shape == (3, 1)


def test_invalid_graphs_fail_early() -> None:
    with pytest.raises(TypeError, match="target must be callable"):
        tr.route(42)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="step must be callable"):
        tr.Sequential(42)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires at least one step"):
        tr.Sequential()
    with pytest.raises(TypeError, match="unexpected keyword"):
        tr.Parallel(copy=lambda x: x)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="unexpected keyword"):
        tr.Sequential(named=lambda x: x)  # type: ignore[call-arg]
