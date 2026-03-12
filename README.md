# Tharvexal: Physics-Informed Optimizer

A custom PyTorch optimizer that uses Newtonian mechanics and Langevin dynamics to navigate loss landscapes.

## Features

- **Physics-based dynamics** with configurable mass, friction, and velocity
- **Momentum** (β) for velocity smoothing
- **Adaptive learning rate** (Adam-like second moment scaling)
- **Weight decay** (L2 regularization)
- **Gradient clipping** for training stability
- **Warmup schedule** for gradual learning rate increase
- **Langevin dynamics** with temperature for stochastic exploration

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from TharvexalPhysicsInformedOptimizerVsAdamVsSgdBenchmark import Tharvexal

model = ...  # your PyTorch model
optimizer = Tharvexal(
    model.parameters(),
    lr=0.05,
    mass=0.5,
    friction=0.2,
    momentum=0.7,
    temperature=0.01,
)

for epoch in range(epochs):
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
```

## Running the Benchmark

```bash
python TharvexalPhysics-InformedOptimizerVsAdamVsSgdBenchmark.py
```

This runs 10 tests comparing Tharvexal against Adam and SGD on tasks including quadratic bowls, Rosenbrock valley, XOR classification, saddle point escape, and more.

## Recommended Hyperparameters

| Parameter     | Range          | Description                          |
|---------------|----------------|--------------------------------------|
| `lr`          | 0.01 – 0.1    | Learning rate / time step            |
| `friction`    | 0.05 – 0.2    | Velocity damping coefficient         |
| `momentum`    | 0.3 – 0.7     | Velocity exponential moving average  |
| `temperature` | 0.0 – 0.1     | 0.0 = deterministic, >0 = stochastic |
| `mass`        | 0.3 – 1.0     | Inertial mass for dynamics           |

## License

See [LICENSE](LICENSE) for details.
