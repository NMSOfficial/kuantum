"""Detector geometry definitions for the Kuantum simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List


@dataclass
class DetectorLayer:
    """A cylindrical detector layer."""

    name: str
    inner_radius: float
    outer_radius: float
    length: float
    material: str
    color: str = "white"

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "inner_radius": self.inner_radius,
            "outer_radius": self.outer_radius,
            "length": self.length,
            "material": self.material,
            "color": self.color,
        }


@dataclass
class DetectorGeometry:
    """Complete detector geometry comprised of multiple cylindrical layers."""

    layers: List[DetectorLayer] = field(default_factory=list)

    def add_layer(self, layer: DetectorLayer) -> None:
        self.layers.append(layer)

    def __iter__(self):
        return iter(self.layers)

    def __len__(self):
        return len(self.layers)

    def summary(self) -> str:
        lines = ["Detector Geometry Summary:"]
        for layer in self.layers:
            lines.append(
                f"- {layer.name}: r=[{layer.inner_radius}, {layer.outer_radius}] m, length={layer.length} m, material={layer.material}"
            )
        return "\n".join(lines)

    def get_layer(self, name: str) -> DetectorLayer:
        for layer in self.layers:
            if layer.name == name:
                return layer
        raise KeyError(f"Unknown detector layer: {name}")


def default_geometry() -> DetectorGeometry:
    """Create a default three-layer detector geometry."""

    geometry = DetectorGeometry(
        layers=[
            DetectorLayer(
                name="Inner Detector",
                inner_radius=0.02,
                outer_radius=1.5,
                length=6.0,
                material="Silicon",
                color="#1f77b4",
            ),
            DetectorLayer(
                name="Calorimeter",
                inner_radius=1.5,
                outer_radius=4.0,
                length=7.0,
                material="Scintillator",
                color="#ff7f0e",
            ),
            DetectorLayer(
                name="Muon System",
                inner_radius=4.0,
                outer_radius=8.0,
                length=10.0,
                material="Drift Tubes",
                color="#2ca02c",
            ),
        ]
    )
    return geometry
