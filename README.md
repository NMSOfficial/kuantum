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
    ├── analysis.py            # Multi-agent analysis panel and metrics
    ├── detector.py             # Parametric detector geometry
    ├── geometry_io.py          # GDML/ROOT import helpers
    ├── physics.py              # Monte Carlo event generation utilities
    ├── reporting.py            # LaTeX/analysis reporting pipeline
    ├── transport.py            # GPU-accelerated transport estimates
    ├── visualization.py        # Optional PyVista based visualisation helpers
    ├── vr.py                   # Three.js/VR scene exporter
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
- `--geometry-file`: load a GDML/ROOT/JSON detector description instead of the default cylindrical mock-up.
- `--export-vr`: write a self-contained Three.js scene (with WebXR support) for immersive playback.
- `--report`: export a LaTeX run report summarising geometry, per-event energies, and model outputs (combine with `--compile-report` to call `pdflatex`).

The CLI preloads every event into a cinematic timeline, prints a per-event summary, and—when visualisation is enabled—renders a detector-inspired scene with silicon tracker barrels, calorimeter volumes, and muon stations. Interaction points inherit analysis-driven colour palettes, labels report transverse momentum/η alongside detector layers, and the HUD tracks the active phase of the collision together with model and truth labels. Events are synthesised from three phenomenological profiles that mirror the pretrained classifier’s labels: **Higgs Boson Candidate**, **Dark Matter Signature**, and **QCD Background**. Each profile drives distinct energy scales, detector channels, and storyboard annotations so the playback mirrors typical collider studies.

The synthesised detector signatures intentionally map to familiar analysis objects instead of abstract placeholders. You will see

- **Central Jet** and **b-tagged Jet** sprays inside the tracker volume for Higgs-like decays,
- **Prompt Photon** flashes and **Tracker Lepton Tracks** for bosonic and electroweak products,
- **MET Recoil** vectors highlighting missing transverse energy in dark-matter scenarios, and
- **Forward/Endcap hadronic showers** that characterise QCD background activity.

These labels carry through to the console logs and the PyVista overlays—now presented with experiment-style typography, axes, and transverse plane guides—so the narrative matches the model’s Higgs/dark-matter/QCD focus.

During playback use the keyboard to control the cinematic stream:

- `+` speeds up the flow, `-` enters slow motion, and `1` resets to real-time.
- `p` (or hitting the space bar inside the 3D window) toggles pause/resume, while `q` gracefully ends the run.
- `b` (or tapping/clicking inside the viewport) arms the next collision for an extended slow-motion “bullet time” reveal of the proton injection, focusing magnets, and energy cascade phases.

## Advanced capabilities

### Real detector geometries
Pass `--geometry-file` with a GDML or ROOT description exported from an experiment (e.g. ATLAS/CMS subsystems). The loader converts tubes and layer tables into the internal `DetectorGeometry`, colour-codes materials, and preserves radii/lengths in metres.

### Geant4-inspired transport approximations
The `TransportSimulator` consumes the generated particles on the GPU (CUDA if available) and integrates approximate path lengths, producing layer-by-layer energy deposition summaries. These appear in the console and drive the analysis agents.

### Multi-agent analysis panel
Three specialised agents—cross-section tallies, transport diagnostics, and anomaly watch—continuously update a metrics panel visible in both the console log and the PyVista HUD. Extend `kuantum/simulation/analysis.py` with additional agents to incorporate trigger studies or rate monitoring.

### Immersive VR exports
With `--export-vr` the timeline is converted into a self-contained Three.js/WebXR scene. Open the HTML file in any modern browser (desktop or headset) to orbit, zoom, or step into the detector while tracks animate around you.

### Physics-grade reporting
Use `--report` to emit a LaTeX document summarising the geometry and per-event energies. Combine with `--compile-report` to run `pdflatex` (if available) and produce a shareable PDF for analysis meetings.

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
