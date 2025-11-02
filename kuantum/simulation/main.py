"""Command line interface for the Kuantum detector simulation."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from ..models import load_model_and_scaler
from .detector import DetectorGeometry, default_geometry
from .physics import CollisionEvent, generate_event_stream
from .visualization import DetectorVisualizer, render_event


@dataclass
class SimulationConfig:
    max_events: Optional[int]
    event_rate: float
    prediction_filter: Optional[int]
    visualize: bool


class EventLoop:
    def __init__(self, geometry: DetectorGeometry, config: SimulationConfig):
        self.geometry = geometry
        self.config = config
        artifacts = load_model_and_scaler()
        self.model = artifacts.model
        self.scaler = artifacts.scaler
        self.device = artifacts.device
        self.num_continuous = artifacts.config.num_continuous

    def run(self) -> None:
        visualizer = DetectorVisualizer(self.geometry) if self.config.visualize else None
        for event in generate_event_stream(self.geometry, max_events=self.config.max_events):
            prediction = self.predict_event(event)
            event.attach_prediction(prediction)
            if self.config.prediction_filter is not None and prediction != self.config.prediction_filter:
                continue
            self.display_event(event)
            if visualizer and visualizer.is_available():
                plotter = render_event(event, visualizer)
                if plotter:
                    plotter.show(auto_close=True)
            if self.config.event_rate > 0:
                time.sleep(1.0 / self.config.event_rate)

    def predict_event(self, event: CollisionEvent) -> int:
        features = event.features
        if features.shape[0] != self.num_continuous:
            features = np.pad(features, (0, max(0, self.num_continuous - features.shape[0])), mode="constant")
            features = features[: self.num_continuous]
        scaled = self.scaler.transform(features.reshape(1, -1))
        x_cont = np.asarray(scaled, dtype=np.float32)
        import torch

        with torch.no_grad():
            tensor = torch.tensor(x_cont, device=self.device)
            x_categ = torch.empty((tensor.size(0), 0), dtype=torch.long, device=self.device)
            x_cont_enc = self.model.encode_continuous(tensor)
            x_categ_enc = self.model.encode_categorical(x_categ)
            logits = self.model(x_categ, tensor, x_categ_enc, x_cont_enc)
            pred = torch.argmax(logits, dim=-1).item()
        return int(pred)

    def display_event(self, event: CollisionEvent) -> None:
        print(f"Event {event.event_id} -> prediction: {event.model_prediction}")
        for particle in event.particles:
            print(
                f"  {particle.name:8s} | E={particle.four_vector.energy:7.2f} GeV | eta={particle.eta:+.2f} | layer={particle.detector_layer}"
            )
        print("-" * 60)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Kuantum detector simulation")
    parser.add_argument("--events", type=int, default=10, help="Number of events to simulate")
    parser.add_argument("--rate", type=float, default=2.0, help="Events per second (0 for as fast as possible)")
    parser.add_argument("--visualize", action="store_true", help="Enable 3D visualization with PyVista")
    parser.add_argument("--filter", type=int, default=None, help="Only display events predicted as the given class label")
    return parser.parse_args(argv)


def run_cli(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    geometry = default_geometry()
    config = SimulationConfig(
        max_events=args.events,
        event_rate=args.rate,
        prediction_filter=args.filter,
        visualize=args.visualize,
    )
    loop = EventLoop(geometry, config)
    try:
        loop.run()
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")


if __name__ == "__main__":
    run_cli(sys.argv[1:])
