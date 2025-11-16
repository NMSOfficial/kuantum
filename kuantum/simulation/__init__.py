"""Simulation package providing detector geometry, physics and visualization."""

from .analysis import MultiAgentAnalysisPanel
from .detector import DetectorGeometry, DetectorLayer, default_geometry
from .geometry_io import load_geometry
from .main import run_cli
from .physics import CLASS_LABELS, CinematicPhase, CollisionEvent, Particle, generate_event_stream
from .reporting import SimulationReporter
from .transport import TransportSimulator
from .vr import VRSceneExporter

__all__ = [
    "DetectorGeometry",
    "DetectorLayer",
    "CollisionEvent",
    "CinematicPhase",
    "Particle",
    "CLASS_LABELS",
    "default_geometry",
    "generate_event_stream",
    "run_cli",
    "load_geometry",
    "TransportSimulator",
    "SimulationReporter",
    "VRSceneExporter",
    "MultiAgentAnalysisPanel",
]
