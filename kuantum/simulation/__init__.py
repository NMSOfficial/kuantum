"""Simulation package providing detector geometry, physics and visualization."""

from .detector import DetectorGeometry, DetectorLayer, default_geometry
from .main import run_cli
from .physics import CollisionEvent, Particle, generate_event_stream

__all__ = [
    "DetectorGeometry",
    "DetectorLayer",
    "CollisionEvent",
    "Particle",
    "default_geometry",
    "generate_event_stream",
    "run_cli",
]
