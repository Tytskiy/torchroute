# torchroute

[![CI](https://github.com/Tytskiy/torchroute/actions/workflows/ci.yml/badge.svg)](https://github.com/Tytskiy/torchroute/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/torchroute.svg)](https://pypi.org/project/torchroute/)
[![Python](https://img.shields.io/pypi/pyversions/torchroute.svg)](https://pypi.org/project/torchroute/)
[![License](https://img.shields.io/pypi/l/torchroute.svg)](https://github.com/Tytskiy/torchroute/blob/main/LICENSE)

**Compose PyTorch modules with structured inputs and branches.**

`torchroute` extends sequential models with explicit argument routing. Use `prev` for the previous step's
output and `batch` for the model's original input.

```python
import torch
import torchroute as tr


batch = {
    "features": torch.randn(32, 128),
    "target": torch.randn(32, 1),
}

model = tr.Model(
    tr.route(torch.nn.Linear(128, 64), tr.batch["features"]),
    torch.nn.ReLU(),
    tr.NamedParallel(
        prediction=torch.nn.Linear(64, 1),
        target=tr.batch["target"],
    ),
    tr.route(
        torch.nn.functional.mse_loss,
        tr.prev["prediction"],
        tr.prev["target"],
    ),
)

loss = model(batch)
```

The result is an ordinary `torch.nn.Module`.

## Installation

```console
pip install torchroute
```

`torchroute` supports Python 3.10–3.14 and PyTorch 2.2 or newer.

## Routing

An `nn.Sequential` step always receives the previous step's output. Torchroute keeps that default:

```python
model = tr.Model(
    tr.route(encoder, tr.batch["features"]),
    torch.nn.ReLU(),
    head,
)
```

Use `tr.route(...)` when a call needs a different value or more than one argument:

```python
tr.route(
    loss,
    prediction=tr.prev["prediction"],
    target=tr.batch["target"],
)
```

References can follow items and attributes:

```python
tr.batch["user"]["profile"].age
tr.prev["encoder_output"]
```

Regular Python values are passed through unchanged:

```python
tr.route(torch.mean, tr.prev, dim=-1)
```

Use `tr.value(...)` when a value should be a complete model step:

```python
tr.Model(tr.value(42))
```

`tr.route(...)` is a factory that returns a non-callable `Route` specification. Torchroute containers
materialize these specifications automatically and register their target modules directly. Computed inputs
belong in separate container steps, which keeps every trainable module visible to PyTorch.

## Composition

`NamedParallel` runs several steps with the same input and returns a dictionary:

```python
model = tr.Model(
    tr.route(encoder, tr.batch["features"]),
    tr.NamedParallel(
        logits=classifier,
        embedding=projection,
    ),
)
```

`Parallel` is the positional form and returns a tuple:

```python
tr.Parallel(branch_a, branch_b)
```

`Sum` and `Concat` provide common branch reductions. A residual block can be written as:

```python
block = tr.Sum(
    torch.nn.Identity(),
    torch.nn.Sequential(
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
    ),
)
```

Nested containers can build larger structures while sharing the original batch:

```python
user_tower = tr.Sequential(
    tr.route(user_embedding, tr.batch["user_id"]),
    user_encoder,
)

item_tower = tr.Sequential(
    tr.route(item_embedding, tr.batch["item_id"]),
    item_encoder,
)

model = tr.Model(
    tr.NamedParallel(user=user_tower, item=item_tower),
    tr.route(
        loss,
        user=tr.prev["user"],
        item=tr.prev["item"],
        target=tr.batch["target"],
    ),
)
```

## `.route(...)` syntax

Subclass `tr.Module` to get a stable `.route(...)` method:

```python
class MyLayer(tr.Module):
    def forward(self, x, mask): ...


spec = MyLayer().route(
    x=tr.prev,
    mask=tr.batch["mask"],
)
```

The factory form accepts existing modules and arbitrary callables as targets:

```python
spec = tr.route(torch.nn.Linear(128, 64), tr.prev)
```

A route can be materialized when it needs to live outside a torchroute container:

```python
routed_module = spec.as_module()
output = routed_module(x, batch=batch)
```

The method can also be enabled for every PyTorch module:

```python
tr.enable_module_routes()
spec = torch.nn.Linear(128, 64).route(tr.prev)
tr.disable_module_routes()
```

This only adds the method; normal module calls keep their PyTorch behavior.

## PyTorch integration

Torchroute containers use the regular PyTorch ownership tree:

```python
model.parameters()
model.train()
model.eval()
model.to(device)
```

Route specifications do not add wrapper levels to parameter names or checkpoint keys:

```python
model = tr.Model(
    tr.route(torch.nn.Linear(128, 64), tr.batch["features"]),
)

assert list(model.state_dict()) == ["0.weight", "0.bias"]
assert list(dict(model.named_parameters())) == ["0.weight", "0.bias"]
```

Checkpoints from ordinary PyTorch models load directly when their module paths and tensor shapes match. For
example, a matching `nn.Sequential` and `tr.Model` use the same numeric paths. Different paths such as
`encoder.weight` and `0.weight` still require an explicit rename, following the usual PyTorch rules.

The same ownership model works with FSDP auto-wrapping and FSDP2 distributed checkpoints.

## API

```python
tr.Model(...)
tr.Sequential(...)

tr.Parallel(...)
tr.NamedParallel(...)
tr.Sum(...)
tr.Concat(...)

spec = tr.route(target, ...)
spec = module.route(...)  # tr.Module subclasses
spec.as_module()

tr.enable_module_routes()
tr.disable_module_routes()

tr.prev
tr.batch
tr.value(...)
```

## Status

`torchroute` is alpha software. The public API may change before 1.0. Feedback and bug reports are welcome.

## Development

```console
uv sync --dev
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
uv build
```

## License

MIT
