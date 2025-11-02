# Kuantum Simulation Toolkit

This repository provides a modular Monte Carlo event simulation pipeline for the Kuantum detector model. It includes tools for generating synthetic collision events, running the pretrained TabAttention model, and optionally visualising detector geometry in 3D.

## Project structure

```
kuantum/
├── models/
│   ├── checkpoints/            # Saved model weights and scaler
│   ├── config.yaml             # Paths and hyper-parameters for the model
│   ├── loaders.py              # Utilities to instantiate the model and scaler
│   └── tab_attention.py        # Implementation of the TabAttention network
└── simulation/
    ├── detector.py             # Parametric detector geometry
    ├── physics.py              # Monte Carlo event generation utilities
    ├── visualization.py        # Optional PyVista based visualisation helpers
    └── main.py                 # Command line interface and event loop
```

The top-level `predict.py` module exposes a simple API for running predictions and can also launch the simulation CLI.

## Installation

1. Create and activate a Python 3.9+ environment.
2. Install the dependencies:

```bash
pip install torch numpy pyvista pytest einops pyyaml
```

The visualisation step requires PyVista; if it is not installed, the simulation will run in text-only mode.

## Running the simulation

To launch the event loop from the command line:

```bash
python predict.py --events 5 --rate 1 --visualize
```

Command line options:

- `--events`: number of events to generate (default 10).
- `--rate`: target events per second (0 disables throttling).
- `--visualize`: enable real-time 3D rendering with PyVista.
- `--filter`: only display events that the model predicts as a specific class label.

The CLI prints a per-event summary and, when visualisation is enabled, renders detector layers with particle tracks and the predicted class label overlay.

## Programmatic usage

```python
from predict import load_predictor
from kuantum.simulation.physics import generate_event
from kuantum.simulation.detector import default_geometry

predictor = load_predictor()
geometry = default_geometry()
event = generate_event(0, geometry)
label = predictor.predict(event.features)
print(label)
```

## Tests

Run the automated test suite with:

```bash
pytest
```

## License

This project is distributed for research and educational use.
