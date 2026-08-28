from __future__ import annotations

from collections.abc import Iterator
from functools import partial

import pytest
import torch
import torch.distributed as dist

import torchroute as tr


@pytest.fixture(scope="module")
def distributed_process_group(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Gloo process groups are unavailable")

    rendezvous = tmp_path_factory.mktemp("distributed") / "rendezvous"
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def _nested_routed_model() -> tr.Model:
    return tr.Model(
        tr.route(torch.nn.Linear(4, 4), tr.batch["x"]),
        tr.NamedParallel(
            left=torch.nn.Linear(4, 2),
            right=torch.nn.Linear(4, 2),
        ),
        tr.route(torch.add, tr.prev["left"], tr.prev["right"]),
    )


@pytest.mark.filterwarnings("ignore:FSDP is switching to use")
@pytest.mark.filterwarnings("ignore:When using.*NO_SHARD")
def test_fsdp1_checkpoint_round_trip(distributed_process_group: None) -> None:
    del distributed_process_group
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    source = FSDP(
        tr.Model(tr.route(torch.nn.Linear(2, 1), tr.batch["x"])),
        device_id=torch.device("cpu"),
        use_orig_params=True,
    )
    target = FSDP(
        tr.Model(tr.route(torch.nn.Linear(2, 1), tr.batch["x"])),
        device_id=torch.device("cpu"),
        use_orig_params=True,
    )

    source({"x": torch.ones(3, 2)}).sum().backward()
    state = source.state_dict()
    incompatible = target.load_state_dict(state, strict=True)

    assert list(state) == ["0.weight", "0.bias"]
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys


@pytest.mark.filterwarnings("ignore:FSDP is switching to use")
@pytest.mark.filterwarnings("ignore:When using.*NO_SHARD")
def test_fsdp1_auto_wraps_nested_routes_and_trains(distributed_process_group: None) -> None:
    del distributed_process_group
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

    auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=1)
    source = FSDP(
        _nested_routed_model(),
        auto_wrap_policy=auto_wrap_policy,
        device_id=torch.device("cpu"),
        use_orig_params=True,
    )
    target = FSDP(
        _nested_routed_model(),
        auto_wrap_policy=auto_wrap_policy,
        device_id=torch.device("cpu"),
        use_orig_params=True,
    )
    batch = {"x": torch.randn(3, 4)}
    before = {name: tensor.clone() for name, tensor in source.state_dict().items()}

    optimizer = torch.optim.Adam(source.parameters(), lr=0.01)
    source(batch).square().mean().backward()
    optimizer.step()

    state = source.state_dict()
    incompatible = target.load_state_dict(state, strict=True)

    assert any(isinstance(module, FSDP) for module in tuple(source.modules())[1:])
    assert list(state) == [
        "0.weight",
        "0.bias",
        "1.left.weight",
        "1.left.bias",
        "1.right.weight",
        "1.right.bias",
    ]
    assert any(not torch.equal(before[name], tensor) for name, tensor in state.items())
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys
    torch.testing.assert_close(source(batch), target(batch))


@pytest.mark.filterwarnings("ignore:FSDP is switching to use")
@pytest.mark.filterwarnings("ignore:When using.*NO_SHARD")
def test_fsdp1_preserves_shared_module_aliases(distributed_process_group: None) -> None:
    del distributed_process_group
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    shared = torch.nn.Linear(2, 2)
    model = FSDP(
        tr.Model(tr.route(shared, tr.batch["x"]), shared),
        device_id=torch.device("cpu"),
        use_orig_params=True,
    )

    model({"x": torch.ones(3, 2)}).sum().backward()

    assert len(tuple(model.parameters())) == 2
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert list(model.state_dict()) == ["0.weight", "0.bias", "1.weight", "1.bias"]


def test_fsdp2_distributed_checkpoint_round_trip(distributed_process_group: None) -> None:
    del distributed_process_group
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    mesh = init_device_mesh("cpu", (1,))
    source = tr.Model(tr.route(torch.nn.Linear(2, 1), tr.batch["x"]))
    target = tr.Model(tr.route(torch.nn.Linear(2, 1), tr.batch["x"]))
    fully_shard(source, mesh=mesh)
    fully_shard(target, mesh=mesh)

    source({"x": torch.ones(3, 2)}).sum().backward()
    state = get_model_state_dict(source)
    incompatible = set_model_state_dict(target, state)

    assert list(state) == ["0.weight", "0.bias"]
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys


def test_fsdp2_round_trips_nested_model_and_optimizer_state(
    distributed_process_group: None,
) -> None:
    del distributed_process_group
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    mesh = init_device_mesh("cpu", (1,))
    source = _nested_routed_model()
    target = _nested_routed_model()
    fully_shard(source.get_submodule("1"), mesh=mesh)
    fully_shard(source, mesh=mesh)
    fully_shard(target.get_submodule("1"), mesh=mesh)
    fully_shard(target, mesh=mesh)

    source_optimizer = torch.optim.Adam(source.parameters(), lr=0.01)
    target_optimizer = torch.optim.Adam(target.parameters(), lr=0.01)
    batch = {"x": torch.randn(3, 4)}
    source(batch).square().mean().backward()
    source_optimizer.step()

    model_state, optimizer_state = get_state_dict(source, source_optimizer)
    incompatible = set_state_dict(
        target,
        target_optimizer,
        model_state_dict=model_state,
        optim_state_dict=optimizer_state,
    )
    expected_names = {
        "0.weight",
        "0.bias",
        "1.left.weight",
        "1.left.bias",
        "1.right.weight",
        "1.right.bias",
    }

    assert set(model_state) == expected_names
    assert set(optimizer_state["state"]) == expected_names
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys
    torch.testing.assert_close(source(batch), target(batch))
