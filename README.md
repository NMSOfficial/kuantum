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
python predict.py --duration 15 --rate 2 --speed 1.0 --visualize
```

Command line options:

- `--duration`: length of the pre-generated simulation in seconds. If omitted, the CLI prompts interactively.
- `--events`: optional hard limit on the number of generated events.
- `--rate`: target events per second during playback (used to build the cinematic timeline).
- `--speed`: initial playback speed multiplier; adjust live with the keyboard shortcuts shown in the console.
- `--visualize`: enable real-time 3D rendering with PyVista.
- `--filter`: only display events that the model predicts as a specific class label.

The CLI preloads every event into a futuristic “video” timeline, prints a per-event summary, and—when visualisation is enabled—renders neon detector layers, glowing interaction points, and HUD overlays that update with the model prediction and phase of the collision. Events are synthesised from three phenomenological profiles that mirror the pretrained classifier’s labels: **Higgs Boson Candidate**, **Dark Matter Signature**, and **QCD Background**. Each profile drives distinct energy scales, detector channels, and cinematic annotations so the playback feels physically purposeful.

During playback use the keyboard to control the cinematic stream:

- `+` speeds up the flow, `-` enters slow motion, and `1` resets to real-time.
- `p` (or hitting the space bar inside the 3D window) toggles pause/resume, while `q` gracefully ends the run.
- `b` (or tapping/clicking inside the viewport) arms the next collision for an extended slow-motion “bullet time” reveal of the proton injection, focusing magnets, and energy cascade phases.

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
