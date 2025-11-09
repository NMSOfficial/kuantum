"""Visualization utilities for detector geometry and events."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from .detector import DetectorGeometry, DetectorLayer
from .physics import CinematicPhase, CollisionEvent

if TYPE_CHECKING:  # pragma: no cover - circular import guard for type checking only
    from .main import PlaybackController

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
    show_overlay: bool = False
    _plotter: Optional["pv.Plotter"] = field(default=None, init=False, repr=False)
    _dynamic_actors: List[object] = field(default_factory=list, init=False, repr=False)
    _hud_actor: Optional[object] = field(default=None, init=False, repr=False)
    _window_initialized: bool = field(default=False, init=False, repr=False)
    _controller: Optional["PlaybackController"] = field(default=None, init=False, repr=False)

    def is_available(self) -> bool:
        return pv is not None

    def initialize_scene(self) -> Optional["pv.Plotter"]:  # type: ignore[name-defined]
        if not self.is_available():
            return None
        if self._plotter is None:
            plotter, background_managed = self._create_plotter()
            self._plotter = plotter
            self._plotter.set_background("black", top="midnightblue")
            with np.errstate(all="ignore"):
                try:
                    self._plotter.enable_anti_aliasing()
                except Exception:
                    pass
                try:
                    self._plotter.enable_eye_dome_lighting()
                except Exception:
                    pass
            self._apply_parallel_projection(self._plotter, enable=False)
            for layer in self.geometry:
                self._add_cylindrical_layer(self._plotter, layer)
            if self.show_overlay:
                self._hud_actor = self._plotter.add_text(
                    "Collision stream ready",
                    position="upper_left",
                    font_size=14,
                    color="white",
                )
            self._register_callbacks(self._plotter)
            self._configure_camera(self._plotter)
            if background_managed:
                self._window_initialized = True
        if not self._window_initialized:
            self._initialize_window(self._plotter)
            self._window_initialized = True
        return self._plotter

    def bind_controller(self, controller: "PlaybackController") -> None:
        self._controller = controller
        if self._plotter is not None:
            self._register_callbacks(self._plotter)

    def animate_event(self, event: CollisionEvent, *, playback_speed: float = 1.0, bullet_time: bool = False) -> None:
        plotter = self.initialize_scene()
        if plotter is None:
            return
        phases = event.cinematic_phases or [
            CinematicPhase(
                name="Standard Playback",
                duration=0.5,
                track_scale=1.0,
                glow_strength=0.4,
                annotation="Canlı veri akışı",
            )
        ]

        for index, phase in enumerate(phases):
            self._clear_dynamic(plotter)
            self._draw_tracks(
                plotter,
                event,
                track_scale=phase.track_scale,
                glow_strength=phase.glow_strength,
                annotation=phase.annotation,
            )
            self._update_overlay(plotter, event, phase, index, len(phases))
            plotter.render()
            self._pump_events(plotter)
            duration = phase.duration
            if bullet_time and phase.name.lower().startswith("collision"):
                duration *= 3.0
            duration = duration / max(playback_speed, 0.125)
            self._sleep_with_events(plotter, max(duration, 0.02))

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

    def _draw_tracks(
        self,
        plotter: "pv.Plotter",
        event: CollisionEvent,
        *,
        track_scale: float = 1.0,
        glow_strength: float = 0.4,
        annotation: Optional[str] = None,
    ) -> None:
        for index, particle in enumerate(event.particles):
            direction = np.array(
                [particle.four_vector.px, particle.four_vector.py, particle.four_vector.pz], dtype=float
            )
            norm = np.linalg.norm(direction) + 1e-9
            direction /= norm
            line_length = 12.0 * track_scale
            points = self._helical_track(direction, line_length)
            color = _futuristic_palette(index)
            spline = pv.Spline(points, n_points=200)
            track_mesh = spline.tube(radius=0.12)
            actor = plotter.add_mesh(
                track_mesh,
                color=color,
                ambient=0.6,
                specular=0.8,
                smooth_shading=True,
            )
            self._dynamic_actors.append(actor)

            highlight = self._compute_highlight(particle.detector_layer, direction)
            if highlight is not None:
                sphere = pv.Sphere(radius=0.25 * (0.5 + glow_strength), center=highlight)
                halo = pv.Sphere(radius=0.5 * (0.5 + glow_strength), center=highlight)
                glow_actor = plotter.add_mesh(
                    halo,
                    color=color,
                    opacity=0.15 + 0.25 * glow_strength,
                    smooth_shading=True,
                    ambient=0.35 + 0.3 * glow_strength,
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
                    f"{particle.display_name.upper()}\n{particle.detector_layer}\nE={particle.four_vector.energy:.1f} GeV"
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

        if annotation and self.show_overlay:
            annotation_actor = plotter.add_text(
                annotation,
                position="lower_left",
                font_size=12,
                color="#8cfbff",
            )
            self._dynamic_actors.append(annotation_actor)

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

    def _update_overlay(
        self,
        plotter: "pv.Plotter",
        event: CollisionEvent,
        phase: CinematicPhase,
        index: int,
        total: int,
    ) -> None:
        if not self.show_overlay:
            return
        overlay_lines = [
            f"EVENT {event.event_id:05d}",
        ]
        if event.model_prediction is not None:
            overlay_lines.append(f"PRED {event.model_prediction}")
        overlay_lines.append(f"PHASE {index + 1}/{total}: {phase.name.upper()}")
        overlay = "\n".join(overlay_lines)

        if self._hud_actor is not None:
            try:
                plotter.remove_actor(self._hud_actor, reset_camera=False)
            except Exception:
                pass
            self._hud_actor = None

        self._hud_actor = plotter.add_text(
            overlay,
            position="upper_left",
            font_size=14,
            color="cyan",
        )

    def _register_callbacks(self, plotter: "pv.Plotter") -> None:
        if self._controller is None:
            return

        def _on_space(*_args, **_kwargs):
            try:
                self._controller.toggle_pause()
            except Exception:
                return

        try:
            plotter.add_key_event("space", lambda: _on_space())
            plotter.add_key_event("b", lambda: self._controller.request_bullet_time())
            plotter.add_key_event("Up", lambda: self._adjust_zoom(plotter, factor=1.2))
            plotter.add_key_event("Down", lambda: self._adjust_zoom(plotter, factor=0.85))
            plotter.add_key_event("r", lambda: self._reset_camera(plotter))
        except Exception:
            # Optional backend support; ignore failures silently
            pass
        try:
            plotter.enable_trackball_style()
        except Exception:
            pass

    def _adjust_zoom(self, plotter: "pv.Plotter", *, factor: float) -> None:
        camera = getattr(plotter, "camera", None)
        if camera is None:
            return
        for method_name in ("Zoom", "zoom"):
            zoom = getattr(camera, method_name, None)
            if callable(zoom):
                try:
                    zoom(factor)
                    return
                except Exception:
                    continue
        try:
            position = np.asarray(camera.position, dtype=float)
            focal = np.asarray(camera.focal_point, dtype=float)
            direction = focal - position
            position = focal - direction * factor
            camera.position = tuple(position.tolist())
        except Exception:
            pass

    def _reset_camera(self, plotter: "pv.Plotter") -> None:
        try:
            plotter.camera_position = [
                (20.0, 15.0, 12.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
            ]
            plotter.reset_camera_clipping_range()
        except Exception:
            pass

    def _helical_track(self, direction: np.ndarray, length: float) -> np.ndarray:
        """Create a light-weight helical trajectory aligned with the particle momentum."""
        turns = 1.5
        steps = 120
        t = np.linspace(0.0, 1.0, steps)
        axis = np.array([0.0, 0.0, 1.0])
        axis_component = np.dot(direction, axis)
        radial_component = direction - axis_component * axis
        radial_norm = np.linalg.norm(radial_component)
        if radial_norm < 1e-6:
            radial_component = np.array([1.0, 0.0, 0.0])
            radial_norm = 1.0
        radial_component /= radial_norm
        orthogonal = np.cross(direction, radial_component)
        orth_norm = np.linalg.norm(orthogonal)
        if orth_norm < 1e-6:
            orthogonal = np.array([0.0, 1.0, 0.0])
            orth_norm = 1.0
        orthogonal /= orth_norm
        axial = direction / (np.linalg.norm(direction) + 1e-9)
        radius = 0.4
        z_extent = length
        path = []
        for param in t:
            angle = 2 * np.pi * turns * param
            offset = (
                np.cos(angle) * radial_component + np.sin(angle) * orthogonal
            ) * radius
            axial_offset = axial * (param * z_extent)
            path.append(offset + axial_offset)
        return np.vstack([[0.0, 0.0, 0.0], *path])

    def _configure_camera(self, plotter: "pv.Plotter") -> None:
        try:
            plotter.camera_position = [
                (20.0, 15.0, 12.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
            ]
            self._apply_parallel_projection(plotter, enable=False)
            plotter.reset_camera_clipping_range()
        except Exception:
            pass

    def _apply_parallel_projection(self, plotter: "pv.Plotter", *, enable: bool) -> None:
        toggle = getattr(plotter, "enable_parallel_projection", None)
        disable = getattr(plotter, "disable_parallel_projection", None)
        if enable:
            if callable(toggle):
                try:
                    toggle(True)
                    return
                except TypeError:
                    try:
                        toggle()
                        return
                    except Exception:
                        pass
        else:
            if callable(toggle):
                try:
                    toggle(False)
                    return
                except TypeError:
                    try:
                        if callable(disable):
                            disable()
                            return
                        toggle()
                        return
                    except Exception:
                        pass
            if callable(disable):
                try:
                    disable()
                except Exception:
                    pass

    def _create_plotter(self) -> tuple["pv.Plotter", bool]:
        """Create an interactive plotter and return whether it manages its own window."""
        background_cls = getattr(pv, "BackgroundPlotter", None)
        if background_cls is not None:
            try:
                plotter = background_cls(window_size=(1400, 900), title="Kuantum Collider")
                try:
                    plotter.app.processEvents()
                except Exception:
                    pass
                return plotter, True
            except Exception:
                pass
        try:
            plotter = pv.Plotter(window_size=(1400, 900))
        except TypeError:
            plotter = pv.Plotter()
            with np.errstate(all="ignore"):
                try:
                    plotter.window_size = (1400, 900)
                except Exception:
                    pass
        return plotter, False

    def _initialize_window(self, plotter: "pv.Plotter") -> None:
        """Start the plotting window in non-blocking mode when possible."""
        with np.errstate(all="ignore"):
            try:
                plotter.show(auto_close=False, interactive_update=True)
                return
            except TypeError:
                pass
            except Exception:
                # Fall back to other strategies below
                pass
        for kwargs in (
            {"auto_close": False, "interactive": False},
            {"auto_close": False},
            {},
        ):
            try:
                plotter.show(**kwargs)
                break
            except TypeError:
                continue
            except Exception:
                continue
        self._pump_events(plotter)

    def _pump_events(self, plotter: "pv.Plotter") -> None:
        """Process GUI events so the window stays responsive."""
        for attr in ("app", "qt_app", "qtapp", "qapp", "_app"):
            app = getattr(plotter, attr, None)
            if app is None:
                continue
            try:
                app.processEvents()
            except Exception:
                continue
        try:
            plotter.update()
        except Exception:
            pass

    def _sleep_with_events(self, plotter: "pv.Plotter", duration: float) -> None:
        """Sleep while intermittently flushing GUI events."""
        end_time = time.time() + max(duration, 0.0)
        while True:
            remaining = end_time - time.time()
            if remaining <= 0:
                break
            self._pump_events(plotter)
            time.sleep(min(0.02, remaining))
        self._pump_events(plotter)
