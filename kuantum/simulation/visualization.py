"""Visualization utilities for detector geometry and events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .detector import DetectorGeometry, DetectorLayer
from .physics import CollisionEvent

try:  # pragma: no cover - optional dependency
    import pyvista as pv
except Exception:  # pragma: no cover - optional dependency
    pv = None


def _futuristic_palette(index: int) -> str:
    palette = ["#00f7ff", "#ff00d4", "#ffe800", "#21ff6b", "#ff6f61"]
    return palette[index % len(palette)]


@dataclass
class DetectorVisualizer:
    geometry: DetectorGeometry
    show_overlay: bool = True
    _plotter: Optional["pv.Plotter"] = field(default=None, init=False, repr=False)
    _dynamic_actors: List[object] = field(default_factory=list, init=False, repr=False)
    _hud_actor: Optional[object] = field(default=None, init=False, repr=False)
    _window_initialized: bool = field(default=False, init=False, repr=False)

    def is_available(self) -> bool:
        return pv is not None

    def initialize_scene(self) -> Optional["pv.Plotter"]:  # type: ignore[name-defined]
        if not self.is_available():
            return None
        if self._plotter is None:
            self._plotter = pv.Plotter(window_size=(1400, 900))
            self._plotter.set_background("black", top="midnightblue")
            self._plotter.enable_anti_aliasing()
            for layer in self.geometry:
                self._add_cylindrical_layer(self._plotter, layer)
            if self.show_overlay:
                self._hud_actor = self._plotter.add_text(
                    "KUANTUM // READY",
                    position="upper_left",
                    font_size=14,
                    color="cyan",
                )
            self._plotter.add_text(
                "Fütüristik Mod", position="upper_right", font_size=12, color="#ff00d4"
            )
        if not self._window_initialized:
            self._plotter.show(auto_close=False, interactive=False)
            self._window_initialized = True
        return self._plotter

    def animate_event(self, event: CollisionEvent) -> None:
        plotter = self.initialize_scene()
        if plotter is None:
            return
        self._clear_dynamic(plotter)
        self._draw_tracks(plotter, event)
        if self.show_overlay and self._hud_actor is not None and event.model_prediction is not None:
            overlay = f"KUANTUM // PREDICTION {event.model_prediction}\nEVENT {event.event_id:05d}"
            try:
                plotter.update_text(self._hud_actor, overlay)
            except Exception:
                self._hud_actor = plotter.add_text(
                    overlay,
                    position="upper_left",
                    font_size=14,
                    color="cyan",
                )
        plotter.render()

    def finalize(self) -> None:
        if self._plotter is not None:
            self._plotter.close()
            self._plotter = None
            self._window_initialized = False
            self._dynamic_actors.clear()
            self._hud_actor = None

    def _add_cylindrical_layer(self, plotter: "pv.Plotter", layer: DetectorLayer) -> None:
        height = layer.length
        radius = (layer.inner_radius + layer.outer_radius) / 2
        tube = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=radius, height=height)
        plotter.add_mesh(
            tube,
            color=layer.color,
            opacity=0.18,
            smooth_shading=True,
            specular=0.5,
            specular_power=30.0,
        )

    def _draw_tracks(self, plotter: "pv.Plotter", event: CollisionEvent) -> None:
        for index, particle in enumerate(event.particles):
            direction = np.array(
                [particle.four_vector.px, particle.four_vector.py, particle.four_vector.pz], dtype=float
            )
            norm = np.linalg.norm(direction) + 1e-9
            direction /= norm
            line_length = 12.0
            points = np.vstack([np.zeros(3), direction * line_length])
            color = _futuristic_palette(index)
            actor = plotter.add_lines(points, color=color, width=4)
            self._dynamic_actors.append(actor)

            highlight = self._compute_highlight(particle.detector_layer, direction)
            if highlight is not None:
                sphere = pv.Sphere(radius=0.25, center=highlight)
                halo = pv.Sphere(radius=0.5, center=highlight)
                glow_actor = plotter.add_mesh(
                    halo,
                    color=color,
                    opacity=0.18,
                    smooth_shading=True,
                    ambient=0.4,
                    specular=1.0,
                )
                sphere_actor = plotter.add_mesh(
                    sphere,
                    color=color,
                    ambient=0.6,
                    diffuse=0.8,
                    specular=1.0,
                    specular_power=80.0,
                    smooth_shading=True,
                )
                self._dynamic_actors.extend([glow_actor, sphere_actor])
                label = (
                    f"{particle.name.upper()}\n{particle.detector_layer}\nE={particle.four_vector.energy:.1f} GeV"
                )
                label_actor = plotter.add_point_labels(
                    [highlight],
                    [label],
                    point_color=color,
                    text_color="white",
                    font_size=12,
                    always_visible=True,
                    shape_color="#111111",
                    fill_shape=True,
                )
                self._dynamic_actors.append(label_actor)

    def _compute_highlight(self, layer_name: str, direction: np.ndarray) -> Optional[np.ndarray]:
        try:
            layer = self.geometry.get_layer(layer_name)
        except KeyError:
            return None
        radial_direction = direction.copy()
        radial_magnitude = np.linalg.norm(radial_direction[:2])
        if radial_magnitude < 1e-6:
            return np.array([0.0, 0.0, layer.length / 2.0])
        radius = (layer.inner_radius + layer.outer_radius) / 2
        scale = radius / radial_magnitude
        point = radial_direction * scale
        max_z = layer.length / 2
        point[2] = np.clip(point[2], -max_z, max_z)
        return point

    def _clear_dynamic(self, plotter: "pv.Plotter") -> None:
        for actor in self._dynamic_actors:
            with np.errstate(all="ignore"):
                try:
                    plotter.remove_actor(actor, reset_camera=False)
                except Exception:
                    continue
        self._dynamic_actors.clear()
