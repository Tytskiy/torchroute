# torchroute

[![CI](https://github.com/Tytskiy/torchroute/actions/workflows/ci.yml/badge.svg)](https://github.com/Tytskiy/torchroute/actions/workflows/ci.yml)

Declarative data routing for PyTorch modules.

`torchroute` extends sequential module composition with structured inputs,
branches, and explicit argument routing. Routed calls are ordinary
`torch.nn.Module` objects and participate naturally in the PyTorch module tree.

> `torchroute` is currently alpha software. Its core design is usable, but the
> public API may still change before 1.0.

## Installation

Until the first package-index release, install directly from GitHub:

```console
pip install "torchroute @ git+https://github.com/Tytskiy/torchroute.git"
```

## Usage

```python
import torch
import torchroute as tr


class Loss(tr.Module):
    def forward(self, prediction, target):
        return torch.nn.functional.mse_loss(prediction, target)


model = tr.Model(
    tr.route(torch.nn.Linear(128, 64), tr.batch["features"]),
    torch.nn.ReLU(),
    tr.NamedParallel(
        prediction=torch.nn.Linear(64, 1),
        target=tr.batch["target"],
    ),
    Loss().route(
        prediction=tr.prev["prediction"],
        target=tr.prev["target"],
    ),
)

output = model(batch)
```

A route owns its target module and can also be executed directly when both
routing-context values are available:

```python
routed = layer.route(tr.prev)
output = routed(prev=x, batch=batch)
```

## Checkpoints

Routing wrappers are transparent in `state_dict()` and `load_state_dict()`:

```python
model = tr.Model(
    tr.route(torch.nn.Linear(128, 64), tr.batch["features"]),
    torch.nn.ReLU(),
)

assert list(model.state_dict()) == ["0.weight", "0.bias"]
```

Trainable computations nested inside route arguments use the reserved
`_inputs.N` namespace. The actual PyTorch ownership tree remains visible to
module introspection such as `named_parameters()`.

## Module route syntax

The `route(module, ...)` function works with every callable. Native
`.route(...)` syntax can be enabled explicitly for all PyTorch modules:

```python
tr.enable_module_routes()

model = tr.Model(
    torch.nn.Linear(128, 64).route(tr.batch["features"]),
    torch.nn.ReLU(),
)
```

Use `tr.disable_module_routes()` to remove the method again.

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
