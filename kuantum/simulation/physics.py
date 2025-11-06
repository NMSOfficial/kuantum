"""Monte Carlo physics utilities for generating collision events."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Generator, Iterable, List, Optional

import numpy as np

from .detector import DetectorGeometry


@dataclass
class FourVector:
    energy: float
    px: float
    py: float
    pz: float

    def as_array(self) -> np.ndarray:
        return np.array([self.energy, self.px, self.py, self.pz], dtype=np.float32)


@dataclass
class Particle:
    name: str
    mass: float
    charge: int
    four_vector: FourVector
    eta: float
    phi: float
    detector_layer: str

    def to_feature_vector(self) -> np.ndarray:
        return np.array(
            [
                self.four_vector.energy,
                np.linalg.norm([self.four_vector.px, self.four_vector.py, self.four_vector.pz]),
                self.eta,
            ],
            dtype=np.float32,
        )


@dataclass
class CinematicPhase:
    """Storyboard element for rendering a cinematic collision."""

    name: str
    duration: float
    track_scale: float
    glow_strength: float
    annotation: str


@dataclass
class CollisionEvent:
    event_id: int
    particles: List[Particle]
    features: np.ndarray
    model_prediction: Optional[int] = None
    cinematic_phases: List[CinematicPhase] = field(default_factory=list)

    def attach_prediction(self, label: int) -> None:
        self.model_prediction = label


PARTICLE_CATALOG = [
    ("electron", 0.000511, -1),
    ("muon", 0.105, -1),
    ("pion", 0.139, 1),
    ("kaon", 0.494, 1),
    ("proton", 0.938, 1),
    ("photon", 0.0, 0),
]


def sample_energy(scale: float = 50.0) -> float:
    return np.random.exponential(scale)


def sample_pseudorapidity(max_abs_eta: float = 2.5) -> float:
    return np.random.uniform(-max_abs_eta, max_abs_eta)


def sample_phi() -> float:
    return np.random.uniform(-math.pi, math.pi)


def sample_momentum(energy: float, mass: float) -> float:
    return math.sqrt(max(energy**2 - mass**2, 0.0))


def choose_detector_layer(eta: float, geometry: DetectorGeometry) -> str:
    abs_eta = abs(eta)
    if abs_eta < 1.5:
        return geometry.layers[0].name
    if abs_eta < 2.5:
        return geometry.layers[1].name
    return geometry.layers[-1].name


def generate_particle(geometry: DetectorGeometry) -> Particle:
    name, mass, charge = random.choice(PARTICLE_CATALOG)
    energy = sample_energy()
    momentum = sample_momentum(energy, mass)
    eta = sample_pseudorapidity()
    phi = sample_phi()
    px = momentum * math.cos(phi)
    py = momentum * math.sin(phi)
    pz = momentum * math.sinh(eta)
    layer = choose_detector_layer(eta, geometry)
    four_vector = FourVector(energy=energy, px=px, py=py, pz=pz)
    return Particle(name=name, mass=mass, charge=charge, four_vector=four_vector, eta=eta, phi=phi, detector_layer=layer)


def _pad_features(features: np.ndarray, target_length: int) -> np.ndarray:
    if features.size >= target_length:
        return features[:target_length]
    padding = np.zeros(target_length - features.size, dtype=np.float32)
    return np.concatenate([features, padding])


def build_feature_vector(particles: List[Particle], max_particles: int = 11) -> np.ndarray:
    stats = []
    for particle in particles[:max_particles]:
        stats.append(particle.to_feature_vector())
    flat = np.concatenate(stats, axis=0) if stats else np.zeros(0, dtype=np.float32)
    return _pad_features(flat, max_particles * 3)


def build_cinematic_phases(event: CollisionEvent, geometry: DetectorGeometry) -> List[CinematicPhase]:
    total_energy = sum(p.four_vector.energy for p in event.particles)
    energy_scale = np.clip(total_energy / 250.0, 0.4, 1.8)
    focus_layer = max(event.particles, key=lambda p: abs(p.eta)).detector_layer if event.particles else geometry.layers[0].name

    return [
        CinematicPhase(
            name="Injection Sequencer",
            duration=0.35 * energy_scale,
            track_scale=0.15,
            glow_strength=0.1,
            annotation="Proton demetleri halka hatlarına giriyor.",
        ),
        CinematicPhase(
            name="Magnetic Focusing",
            duration=0.45 * energy_scale,
            track_scale=0.35,
            glow_strength=0.25,
            annotation=f"Manyetik lensler ışını {focus_layer} yönünde sıkıştırıyor.",
        ),
        CinematicPhase(
            name="Pre-Collision Drift",
            duration=0.60 * energy_scale,
            track_scale=0.65,
            glow_strength=0.4,
            annotation="Karşıt demetler hizalanıyor, hız kritik seviyeye ulaşıyor.",
        ),
        CinematicPhase(
            name="Collision Apex",
            duration=0.85 * energy_scale,
            track_scale=1.05,
            glow_strength=1.0,
            annotation="Zaman bükülüyor, enerji patlaması gerçekleşiyor!",
        ),
        CinematicPhase(
            name="Cascade Afterglow",
            duration=0.55 * energy_scale,
            track_scale=1.2,
            glow_strength=0.6,
            annotation="Parçacık izleri dedektör katmanlarına yağmur gibi düşüyor.",
        ),
        CinematicPhase(
            name="Thermal Dissipation",
            duration=0.40 * energy_scale,
            track_scale=0.45,
            glow_strength=0.15,
            annotation="Kalorimetreler enerjiyi emiyor, sistem soğuyor.",
        ),
    ]


def generate_event(event_id: int, geometry: DetectorGeometry, max_particles: int = 11) -> CollisionEvent:
    num_particles = np.random.poisson(lam=6) + 1
    particles = [generate_particle(geometry) for _ in range(num_particles)]
    features = build_feature_vector(particles, max_particles=max_particles)
    event = CollisionEvent(event_id=event_id, particles=particles, features=features)
    event.cinematic_phases = build_cinematic_phases(event, geometry)
    return event


def generate_event_stream(
    geometry: DetectorGeometry,
    *,
    max_events: Optional[int] = None,
    max_particles: int = 11,
) -> Generator[CollisionEvent, None, None]:
    event_id = 0
    while max_events is None or event_id < max_events:
        yield generate_event(event_id, geometry, max_particles=max_particles)
        event_id += 1
