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


def test_ref_errors_include_the_failing_path() -> None:
    with pytest.raises(KeyError) as caught:
        tr.batch["user"]["id"].resolve(prev=None, batch={})

    context = getattr(caught.value, "__notes__", caught.value.args)
    assert context[-1] == "while resolving batch['user']['id'] at ['user']"


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        (tr.prev + 3, 10),
        (3 + tr.prev, 10),
        (tr.prev - 3, 4),
        (10 - tr.prev, 3),
        (tr.prev * 3, 21),
        (3 * tr.prev, 21),
        (tr.prev / 2, 3.5),
        (14 / tr.prev, 2),
        (tr.prev // 3, 2),
        (20 // tr.prev, 2),
        (tr.prev % 3, 1),
        (10 % tr.prev, 3),
        (tr.prev**2, 49),
        (2**tr.prev, 128),
        (-tr.prev, -7),
        (+tr.prev, 7),
        (abs(tr.prev - 10), 3),
    ],
)
def test_refs_support_arithmetic(expression: tr.Ref, expected: int | float) -> None:
    assert expression.resolve(prev=7, batch={}) == expected


def test_ref_arithmetic_composes_in_containers_and_preserves_gradients() -> None:
    classification_loss = torch.tensor(2.0, requires_grad=True)
    regression_loss = torch.tensor(5.0, requires_grad=True)
    model = tr.Model(
        tr.NamedParallel(
            classification_loss=tr.batch["classification_loss"],
            regression_loss=tr.batch["regression_loss"],
        ),
        tr.Sum(
            0.8 * tr.prev["classification_loss"],
            0.2 * tr.prev["regression_loss"],
        ),
    )

    loss = model(
        {
            "classification_loss": classification_loss,
            "regression_loss": regression_loss,
        }
    )
    loss.backward()

    torch.testing.assert_close(loss, torch.tensor(2.6))
    torch.testing.assert_close(classification_loss.grad, torch.tensor(0.8))
    torch.testing.assert_close(regression_loss.grad, torch.tensor(0.2))


def test_refs_support_matrix_multiplication() -> None:
    left = torch.tensor([[1.0, 2.0]])
    right = torch.tensor([[3.0], [4.0]])

    result = (tr.batch["left"] @ tr.batch["right"]).resolve(
        prev=None,
        batch={"left": left, "right": right},
    )
    reflected_result = (left @ tr.batch["right"]).resolve(prev=None, batch={"right": right})

    torch.testing.assert_close(result, torch.tensor([[11.0]]))
    torch.testing.assert_close(reflected_result, torch.tensor([[11.0]]))


def test_ref_expression_repr_preserves_structure() -> None:
    expression = 0.8 * tr.prev["classification_loss"] + abs(tr.batch["offset"])

    assert repr(expression) == "((0.8 * prev['classification_loss']) + abs(batch['offset']))"


def test_ref_expressions_reject_hidden_computations() -> None:
    with pytest.raises(TypeError, match="cannot contain a route"):
        tr.prev + tr.route(lambda value: value, tr.prev)
    with pytest.raises(TypeError, match=r"cannot contain an nn\.Module"):
        tr.prev * {"module": torch.nn.Linear(2, 2)}


def test_module_route_and_route_function_have_the_same_semantics() -> None:
    class Add(tr.Module):
        def forward(self, left: int, right: int, scale: int = 1) -> int:
            return (left + right) * scale

    routed_method = Add().route(left=tr.batch["left"], right=tr.batch["right"], scale=2)
    routed_function = tr.route(Add(), left=tr.batch["left"], right=tr.batch["right"], scale=2)

    for routed in (routed_method, routed_function):
        model = tr.Model(routed)
        assert model({"left": 3, "right": 4}) == 14


def test_target_can_be_used_as_a_routed_keyword_argument() -> None:
    class Loss(tr.Module):
        def forward(self, prediction: int, target: int) -> int:
            return prediction - target

    routed_method = Loss().route(prediction=tr.batch["prediction"], target=tr.batch["target"])
    routed_function = tr.route(
        lambda *, prediction, target: prediction - target,
        prediction=tr.batch["prediction"],
        target=tr.batch["target"],
    )

    for routed in (routed_method, routed_function):
        assert tr.Model(routed)({"prediction": 7, "target": 2}) == 5


def test_route_is_a_non_callable_specification_with_explicit_materialization() -> None:
    linear = torch.nn.Linear(2, 1)
    routed = tr.route(linear, tr.prev)
    x = torch.ones(3, 2, requires_grad=True)

    with pytest.raises(TypeError, match="not callable"):
        routed(prev=x, batch={})  # type: ignore[operator]

    materialized = routed.as_module()
    result = materialized(x, batch={})
    result.sum().backward()

    assert result.shape == (3, 1)
    assert not isinstance(routed, torch.nn.Module)
    assert isinstance(materialized, torch.nn.Module)
    assert routed.target is linear
    assert materialized[0] is linear
    assert list(materialized.state_dict()) == ["0.weight", "0.bias"]
    assert linear.weight.grad is not None


def test_model_is_a_batch_only_sequential_root() -> None:
    model = tr.Model(
        tr.route(lambda x: x + 1, tr.batch["x"]),
        lambda x: x * 2,
        lambda x: x - 3,
    )

    assert model({"x": 10}) == 19


def test_model_is_a_batch_only_sequence() -> None:
    def first(x: int) -> int:
        return x + 1

    def second(x: int) -> int:
        return x * 2

    model = tr.Model(tr.route(first, tr.batch["x"]), second)

    assert len(model) == 2
    assert model[0] is first
    assert model[1] is second
    assert model({"x": 4}) == 10


def test_nested_routes_are_rejected() -> None:
    inner = tr.route(
        lambda left, right: left + right,
        tr.batch["left"],
        tr.batch["right"],
    )

    with pytest.raises(TypeError, match="cannot contain another route"):
        tr.route(lambda values: values["result"], {"result": inner})


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
        ("0.weight", linear.weight),
        ("0.bias", linear.bias),
    ]
    assert model[0] is linear

    model.eval()
    assert not model.training
    assert not model[0].training
    assert not linear.training

    model.to(dtype=torch.float64)
    assert linear.weight.dtype == torch.float64


def test_trainable_computations_are_explicit_container_steps() -> None:
    inner = torch.nn.Linear(2, 2)
    outer = torch.nn.Linear(2, 1)
    model = tr.Model(tr.route(inner, tr.batch["x"]), outer)

    assert list(model.state_dict()) == [
        "0.weight",
        "0.bias",
        "1.weight",
        "1.bias",
    ]
    assert model({"x": torch.ones(3, 2)}).shape == (3, 1)

    clone = tr.Model(
        tr.route(torch.nn.Linear(2, 2), tr.batch["x"]),
        torch.nn.Linear(2, 1),
    )
    assert not clone.load_state_dict(model.state_dict()).missing_keys
    assert torch.equal(
        model({"x": torch.ones(3, 2)}),
        clone({"x": torch.ones(3, 2)}),
    )


def test_plain_sequential_checkpoint_loads_into_matching_routed_model() -> None:
    plain = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 2),
    )
    routed = tr.Model(
        tr.route(torch.nn.Linear(3, 4), tr.batch["x"]),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 2),
    )

    incompatible = routed.load_state_dict(plain.state_dict(), strict=True)
    inputs = torch.randn(5, 3)

    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys
    torch.testing.assert_close(routed({"x": inputs}), plain(inputs))


def test_checkpoint_compatibility_follows_pytorch_ownership_paths() -> None:
    class Plain(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(3, 2)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return cast(torch.Tensor, self.encoder(inputs))

    plain = Plain()
    routed = tr.Model(tr.route(torch.nn.Linear(3, 2), tr.batch["x"]))

    incompatible = routed.load_state_dict(plain.state_dict(), strict=False)

    assert incompatible.missing_keys == ["0.weight", "0.bias"]
    assert incompatible.unexpected_keys == ["encoder.weight", "encoder.bias"]


def test_shared_modules_follow_normal_pytorch_aliasing() -> None:
    shared = torch.nn.Linear(2, 2)
    model = tr.Model(
        tr.route(shared, tr.batch["x"]),
        tr.route(shared, tr.prev),
    )

    assert model[0] is shared
    assert model[1] is shared
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
    target_batch_norm = cast(torch.nn.BatchNorm1d, target[0])
    source_batch_norm = cast(torch.nn.BatchNorm1d, source[0])
    assert target_batch_norm.running_mean is not None
    assert source_batch_norm.running_mean is not None
    assert torch.equal(target_batch_norm.running_mean, source_batch_norm.running_mean)


def test_route_can_be_explicitly_materialized_inside_an_ordinary_module() -> None:
    class Wrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = tr.route(torch.nn.Linear(2, 1), tr.prev).as_module()

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return cast(torch.Tensor, self.encoder(prev=value))

    source = Wrapper()
    state = source.state_dict()
    target = Wrapper()

    assert list(state) == ["encoder.0.weight", "encoder.0.bias"]
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


def test_route_arguments_reject_hidden_modules() -> None:
    nested = tr.route(torch.nn.Linear(2, 2), tr.batch["x"])
    with pytest.raises(TypeError, match="cannot contain another route"):
        tr.route(torch.add, nested, tr.prev)
    with pytest.raises(TypeError, match=r"cannot contain an nn\.Module"):
        tr.route(lambda module: module, torch.nn.Linear(2, 2))


def test_model_supports_torch_compile() -> None:
    model = tr.Model(
        tr.route(torch.nn.Linear(2, 1), tr.batch["x"]),
        0.5 * tr.prev,
    )
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
