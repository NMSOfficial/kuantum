"""Visualization utilities for detector geometry and events."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from .detector import DetectorGeometry, DetectorLayer
from .physics import CollisionEvent

try:  # pragma: no cover - optional dependency
    import pyvista as pv
except Exception:  # pragma: no cover - optional dependency
    pv = None


@dataclass
class DetectorVisualizer:
    geometry: DetectorGeometry
    show_overlay: bool = True

    def is_available(self) -> bool:
        return pv is not None

    def plot_geometry(self) -> Optional[pv.Plotter]:  # type: ignore[name-defined]
        if not self.is_available():
            return None
        plotter = pv.Plotter()
        for layer in self.geometry:
            self._add_cylindrical_layer(plotter, layer)
        if self.show_overlay:
            plotter.add_text("Kuantum Detector", font_size=12)
        return plotter

    def _add_cylindrical_layer(self, plotter: "pv.Plotter", layer: DetectorLayer) -> None:
        height = layer.length
        radius = (layer.inner_radius + layer.outer_radius) / 2
        tube = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=radius, height=height)
        plotter.add_mesh(tube, color=layer.color, opacity=0.2)

    def add_event(self, plotter: "pv.Plotter", event: CollisionEvent) -> None:
        if not self.is_available():  # pragma: no cover - optional dependency
            return
        for particle in event.particles:
            p0 = np.array([0.0, 0.0, 0.0])
            direction = np.array([particle.four_vector.px, particle.four_vector.py, particle.four_vector.pz])
            direction = direction / (np.linalg.norm(direction) + 1e-9)
            points = np.vstack([p0, direction * 10.0])
            plotter.add_lines(points, color="white")
        if event.model_prediction is not None and self.show_overlay:
            plotter.add_text(f"Prediction: {event.model_prediction}", position="upper_left", font_size=14)


def render_event(event: CollisionEvent, visualizer: DetectorVisualizer) -> Optional[pv.Plotter]:  # type: ignore[name-defined]
    if not visualizer.is_available():
        return None
    plotter = visualizer.plot_geometry()
    if plotter is None:
        return None
    visualizer.add_event(plotter, event)
    return plotter
