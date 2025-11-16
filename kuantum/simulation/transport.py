"""GPU-accelerated particle transport approximations for the detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, TYPE_CHECKING

import torch

from .detector import DetectorGeometry

if TYPE_CHECKING:
    from .physics import Particle


_MATERIAL_DENSITY_MAP: Dict[str, float] = {
    "Silicon": 2.33,  # g/cm^3
    "Scintillator": 1.03,
    "Drift Tubes": 0.001225,
    "Lead": 11.34,
    "Argon": 1.40,
    "Aluminium": 2.70,
}


@dataclass
class EnergyDeposit:
    particle_index: int
    layer_name: str
    path_length: float
    energy_loss: float


@dataclass
class TransportSummary:
    deposits: List[EnergyDeposit]
    layer_totals: Dict[str, float]


class TransportSimulator:
    """Approximate particle transport through cylindrical detector layers."""

    def __init__(self, geometry: DetectorGeometry, *, device: Optional[torch.device] = None) -> None:
        self.geometry = geometry
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def propagate(self, particles: Iterable["Particle"]) -> TransportSummary:
        particle_list = list(particles)
        if not particle_list:
            return TransportSummary(deposits=[], layer_totals={})

        theta = self._polar_angles(particle_list)
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        energies = torch.tensor(
            [particle.four_vector.energy for particle in particle_list], dtype=torch.float32, device=self.device
        )

        deposits: List[EnergyDeposit] = []
        layer_totals: Dict[str, float] = {}
        for layer_index, layer in enumerate(self.geometry.layers):
            path_length = self._path_length_through_layer(layer_index, sin_theta, cos_theta)
            energy_loss = self._estimate_energy_loss(layer.material, energies, path_length)
            layer_loss = float(torch.sum(energy_loss).item())
            if layer_loss <= 0:
                continue
            layer_totals[layer.name] = layer_totals.get(layer.name, 0.0) + layer_loss
            for particle_index, (length_value, loss_value) in enumerate(zip(path_length.cpu().numpy(), energy_loss.cpu().numpy())):
                if length_value <= 0 or loss_value <= 0:
                    continue
                deposits.append(
                    EnergyDeposit(
                        particle_index=particle_index,
                        layer_name=layer.name,
                        path_length=float(length_value),
                        energy_loss=float(loss_value),
                    )
                )
        return TransportSummary(deposits=deposits, layer_totals=layer_totals)

    def _polar_angles(self, particles: List["Particle"]) -> torch.Tensor:
        eta = torch.tensor([particle.eta for particle in particles], dtype=torch.float32, device=self.device)
        theta = 2.0 * torch.atan(torch.exp(-eta))
        return theta

    def _path_length_through_layer(self, layer_index: int, sin_theta: torch.Tensor, cos_theta: torch.Tensor) -> torch.Tensor:
        layer = self.geometry.layers[layer_index]
        half_length = layer.length / 2.0
        sin_safe = torch.clamp(sin_theta.abs(), min=1e-4)
        cos_abs = torch.abs(cos_theta)

        inner = torch.tensor(layer.inner_radius, dtype=torch.float32, device=self.device)
        outer = torch.tensor(layer.outer_radius, dtype=torch.float32, device=self.device)

        s_enter = inner / sin_safe
        s_exit = outer / sin_safe
        max_s = half_length / (cos_abs + 1e-4)
        s_exit = torch.minimum(s_exit, max_s)
        valid = (s_exit > 0) & (s_exit > s_enter)
        path = torch.where(valid, s_exit - torch.clamp(s_enter, min=0.0), torch.zeros_like(s_exit))
        return path

    def _estimate_energy_loss(
        self, material: str, energies: torch.Tensor, path_length: torch.Tensor
    ) -> torch.Tensor:
        density = _MATERIAL_DENSITY_MAP.get(material, 1.0)
        stopping_power = 1.5e-3 * density  # GeV/mm equivalent
        path_mm = path_length * 1_000.0
        loss = energies * (1.0 - torch.exp(-stopping_power * path_mm))
        return torch.clamp(loss, min=0.0)


__all__ = ["TransportSimulator", "EnergyDeposit", "TransportSummary"]
