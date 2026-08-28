# torchroute

[![CI](https://github.com/Tytskiy/torchroute/actions/workflows/ci.yml/badge.svg)](https://github.com/Tytskiy/torchroute/actions/workflows/ci.yml)

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

## Why?

`nn.Sequential` works well when every module consumes the previous module's output. Models with structured
batches or multiple inputs usually need a custom `forward()`:

```python
def forward(self, batch):
    encoded = self.encoder(batch["features"])
    prediction = self.head(encoded)
    return self.loss(prediction, batch["target"])
```

As models grow, this often becomes repetitive. `torchroute` keeps those connections in the model definition
while the modules themselves stay unchanged.

## How it works

Callables and `nn.Module`s receive the previous step's output automatically:

```python
tr.Model(
    tr.route(encoder, tr.batch["features"]),
    torch.nn.ReLU(),
    head,
)
```

Use `tr.route(...)` when a step needs other arguments:

```python
tr.route(
    loss,
    prediction=tr.prev["prediction"],
    target=tr.batch["target"],
)
```

This is equivalent to calling:

```python
loss(
    prediction=prev["prediction"],
    target=batch["target"],
)
```

References can follow nested items and attributes:

```python
tr.batch["user"]["profile"].age
tr.prev["encoder_output"]
```

They are resolved when the model runs.

## Branches

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

`Sum` and `Concat` provide common branch reductions. For example, a residual block can be written as:

```python
block = tr.Sum(
    torch.nn.Identity(),
    torch.nn.Sequential(
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
    ),
)
```

## Structured batches

Different parts of a model can read from the same structured input:

```python
user_tower = tr.Sequential(
    tr.route(user_embedding, tr.batch["user"]["id"]),
    user_encoder,
)

item_tower = tr.Sequential(
    tr.route(item_embedding, tr.batch["item"]["id"]),
    item_encoder,
)

model = tr.Model(
    tr.NamedParallel(
        user=user_tower,
        item=item_tower,
    ),
    tr.route(
        loss,
        user=tr.prev["user"],
        item=tr.prev["item"],
        target=tr.batch["target"],
    ),
)
```

Routes can also be nested, and regular Python values can be passed directly:

```python
tr.route(outer, tr.route(inner, tr.batch["x"]))
tr.route(torch.mean, tr.prev, dim=-1)
```

Use `tr.value(...)` to turn a value into a model step:

```python
tr.Model(tr.value(42))
```

## Native `.route(...)` syntax

`torchroute` provides a `Module` base class with a `.route(...)` method:

```python
class MyLayer(tr.Module):
    def forward(self, x, mask): ...


routed = MyLayer().route(
    x=tr.prev,
    mask=tr.batch["mask"],
)
```

The function form works with existing PyTorch modules and arbitrary callables:

```python
tr.route(torch.nn.Linear(128, 64), tr.prev)
```

You can also enable the method for all PyTorch modules:

```python
tr.enable_module_routes()
layer = torch.nn.Linear(128, 64).route(tr.prev)
tr.disable_module_routes()
```

This adds the `route` method to `nn.Module` without changing normal module calls.

## PyTorch integration

Routes and containers support regular PyTorch operations such as:

```python
model.parameters()
model.train()
model.eval()
model.to(device)
```

Routing wrappers remain transparent in checkpoints:

```python
model = tr.Model(
    tr.route(torch.nn.Linear(128, 64), tr.batch["features"]),
)

assert list(model.state_dict()) == ["0.weight", "0.bias"]
```

`load_state_dict()` uses the same keys.

## API at a glance

```python
tr.Model(...)
tr.Sequential(...)

tr.Parallel(...)
tr.NamedParallel(...)
tr.Sum(...)
tr.Concat(...)

tr.route(target, ...)
module.route(...)

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
