# torchroute

[![CI](https://github.com/Tytskiy/torchroute/actions/workflows/ci.yml/badge.svg)](https://github.com/Tytskiy/torchroute/actions/workflows/ci.yml)

**Build multi-input and branching PyTorch models without plumbing-heavy `forward()` methods.**

`torchroute` is a small library for composing PyTorch modules with structured inputs, branches,
residual connections, and multiple argument sources. It extends sequential composition with two runtime
references: `prev`, the previous step's output, and `batch`, the model's original input.

```python
import torch
import torchroute as tr


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

The resulting model is an ordinary `torch.nn.Module`. Each route resolves its arguments when the model runs
and then calls its target.

## Installation

Until the first PyPI release, install directly from GitHub:

```console
pip install "torchroute @ git+https://github.com/Tytskiy/torchroute.git"
```

`torchroute` supports Python 3.10–3.14 and PyTorch 2.2 or newer.

## Why?

`nn.Sequential` works well while every module consumes the previous module's output. Structured batches and
multiple argument sources usually move that wiring into `forward()`:

```python
def forward(self, batch):
    encoded = self.encoder(batch["features"])
    prediction = self.head(encoded)
    return self.loss(prediction, batch["target"])
```

That wiring becomes repetitive in models assembled from reusable feature pipelines, towers, losses, and
branches. With `torchroute`, the connections live in the model definition:

```python
tr.Model(
    tr.route(encoder, tr.batch["features"]),
    tr.NamedParallel(
        prediction=head,
        target=tr.batch["target"],
    ),
    tr.route(
        loss,
        prediction=tr.prev["prediction"],
        target=tr.prev["target"],
    ),
)
```

The modules keep their regular PyTorch interfaces; the composition describes how values move between them.

## Runtime references

`tr.prev` refers to the output of the previous step:

```python
tr.route(layer, tr.prev)
```

`tr.batch` refers to the original value passed to `Model`:

```python
tr.route(embedding, tr.batch["user"]["id"])
```

References follow nested items and attributes:

```python
tr.batch["user"]["profile"].age
tr.prev["encoder_output"]
```

A route such as

```python
tr.route(
    loss,
    prediction=tr.prev["prediction"],
    target=tr.batch["target"],
)
```

resolves at runtime to the equivalent call:

```python
loss(
    prediction=prev["prediction"],
    target=batch["target"],
)
```

These references are lightweight path descriptions resolved during regular model execution. They require no
graph construction pass or representative tensors.

## Composition

Ordinary callables and `nn.Module`s receive `prev` automatically, so the simple case remains close to
`nn.Sequential`:

```python
model = tr.Model(
    tr.route(torch.nn.Linear(128, 64), tr.batch["features"]),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 32),
)
```

Use `tr.route(...)` where a step needs explicit arguments.

### Branches

`NamedParallel` runs several steps with the same `prev` and returns a dictionary:

```python
model = tr.Model(
    tr.route(encoder, tr.batch["features"]),
    tr.NamedParallel(
        logits=classifier,
        embedding=projection,
    ),
)
```

`Parallel` provides the positional form and returns a tuple:

```python
tr.Parallel(branch_a, branch_b)
```

`Sum` and `Concat` cover common branch reductions. A residual block can be written as:

```python
block = tr.Sum(
    torch.nn.Identity(),
    torch.nn.Sequential(
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
    ),
)
```

### Structured batches

The input structure stays intact throughout the model. For example, two towers can read different parts of
the same batch:

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

### Nested routes and values

Routes compose inside argument structures:

```python
tr.route(
    outer,
    tr.route(inner, tr.batch["x"]),
)
```

Python values remain regular arguments, and `tr.value(...)` turns one into a model step:

```python
tr.route(torch.mean, tr.prev, dim=-1)
tr.Model(tr.value(42))
```

## Native `.route(...)` syntax

`torchroute` provides a `Module` base class with a stable `.route(...)` method:

```python
class MyLayer(tr.Module):
    def forward(self, x, mask): ...


routed = MyLayer().route(
    x=tr.prev,
    mask=tr.batch["mask"],
)
```

The explicit function works with existing PyTorch modules and arbitrary callables:

```python
tr.route(torch.nn.Linear(128, 64), tr.prev)
```

The same method can be enabled for all PyTorch modules:

```python
tr.enable_module_routes()
layer = torch.nn.Linear(128, 64).route(tr.prev)
```

`enable_module_routes()` adds only the `route` convenience method to `nn.Module`; normal calls continue
through PyTorch's existing `__call__`. Use `tr.disable_module_routes()` to remove the method again.

## PyTorch integration

Routes and containers participate in the regular module lifecycle:

```python
model.parameters()
model.named_parameters()
model.train()
model.eval()
model.to(device)
model.to(dtype=torch.bfloat16)
```

Routing wrappers expose clean checkpoint keys:

```python
model = tr.Model(
    tr.route(torch.nn.Linear(128, 64), tr.batch["features"]),
)

assert list(model.state_dict()) == ["0.weight", "0.bias"]
```

`load_state_dict()` consumes the same representation. Module introspection still shows the actual ownership
path, such as `0.target.weight`, and trainable routes nested inside arguments use the reserved `_inputs.N`
namespace.

The test suite includes a full-graph `torch.compile(..., backend="eager")` smoke test.

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

`torchroute` is alpha software. The core design is usable, while the public API may still change before 1.0.
Feedback, experiments, bug reports, and ideas are welcome.

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
