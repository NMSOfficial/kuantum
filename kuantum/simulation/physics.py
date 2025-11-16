"""Monte Carlo physics utilities for generating collision events."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Sequence, Tuple

import numpy as np

from .detector import DetectorGeometry
from .transport import TransportSimulator, TransportSummary


@dataclass
class FourVector:
    energy: float
    px: float
    py: float
    pz: float

    def as_array(self) -> np.ndarray:
        return np.array([self.energy, self.px, self.py, self.pz], dtype=np.float32)


@dataclass(frozen=True)
class SignalChannel:
    """Template describing how an instrument channel behaves in an event."""

    name: str
    display_label: str
    energy_modifier: float
    eta_center: float
    eta_spread: float
    momentum_fraction: float


@dataclass
class Particle:
    name: str
    four_vector: FourVector
    eta: float
    phi: float
    detector_layer: str
    channel: Optional[SignalChannel] = None

    def to_feature_vector(self) -> np.ndarray:
        return np.array(
            [
                self.four_vector.energy,
                np.linalg.norm([self.four_vector.px, self.four_vector.py, self.four_vector.pz]),
                self.eta,
            ],
            dtype=np.float32,
        )

    @property
    def display_name(self) -> str:
        if self.channel is not None:
            return self.channel.display_label
        return self.name.replace("_", " ").title()


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
    true_label: Optional[int] = None
    event_family: Optional[str] = None
    cinematic_phases: List[CinematicPhase] = field(default_factory=list)
    transport_summary: Optional[TransportSummary] = None

    def attach_prediction(self, label: int) -> None:
        self.model_prediction = label


CLASS_LABELS: Dict[int, str] = {
    0: "Higgs Boson Candidate",
    1: "Dark Matter Signature",
    2: "QCD Background",
}


SIGNAL_CHANNELS: Dict[str, SignalChannel] = {
    "central_jet": SignalChannel(
        name="central_jet",
        display_label="Central Jet",
        energy_modifier=1.15,
        eta_center=0.0,
        eta_spread=0.7,
        momentum_fraction=0.92,
    ),
    "b_tagged_jet": SignalChannel(
        name="b_tagged_jet",
        display_label="b-tagged Jet",
        energy_modifier=1.35,
        eta_center=0.25,
        eta_spread=0.55,
        momentum_fraction=0.88,
    ),
    "tracker_lepton_track": SignalChannel(
        name="tracker_lepton_track",
        display_label="Tracker Lepton Track",
        energy_modifier=0.65,
        eta_center=0.15,
        eta_spread=0.5,
        momentum_fraction=0.75,
    ),
    "prompt_photon": SignalChannel(
        name="prompt_photon",
        display_label="Prompt Photon",
        energy_modifier=0.85,
        eta_center=0.05,
        eta_spread=0.5,
        momentum_fraction=1.05,
    ),
    "missing_transverse_energy": SignalChannel(
        name="missing_transverse_energy",
        display_label="MET Recoil",
        energy_modifier=0.55,
        eta_center=0.0,
        eta_spread=1.15,
        momentum_fraction=1.0,
    ),
    "forward_jet": SignalChannel(
        name="forward_jet",
        display_label="Forward Jet",
        energy_modifier=0.9,
        eta_center=2.0,
        eta_spread=0.45,
        momentum_fraction=0.97,
    ),
    "endcap_hadronic_shower": SignalChannel(
        name="endcap_hadronic_shower",
        display_label="Endcap Hadronic Shower",
        energy_modifier=1.05,
        eta_center=2.3,
        eta_spread=0.55,
        momentum_fraction=1.02,
    ),
}


@dataclass(frozen=True)
class EventProfile:
    label: int
    name: str
    selection_weight: float
    base_cluster_rate: float
    energy_scale: float
    eta_spread: float
    forward_bias: float
    channel_distribution: Sequence[Tuple[str, float]]


EVENT_PROFILES: Dict[int, EventProfile] = {
    0: EventProfile(
        label=0,
        name=CLASS_LABELS[0],
        selection_weight=0.35,
        base_cluster_rate=6.5,
        energy_scale=70.0,
        eta_spread=0.85,
        forward_bias=0.35,
        channel_distribution=[
            ("b_tagged_jet", 0.32),
            ("central_jet", 0.28),
            ("prompt_photon", 0.16),
            ("tracker_lepton_track", 0.14),
            ("missing_transverse_energy", 0.1),
        ],
    ),
    1: EventProfile(
        label=1,
        name=CLASS_LABELS[1],
        selection_weight=0.25,
        base_cluster_rate=5.0,
        energy_scale=85.0,
        eta_spread=1.15,
        forward_bias=0.55,
        channel_distribution=[
            ("missing_transverse_energy", 0.38),
            ("forward_jet", 0.22),
            ("central_jet", 0.16),
            ("prompt_photon", 0.12),
            ("tracker_lepton_track", 0.12),
        ],
    ),
    2: EventProfile(
        label=2,
        name=CLASS_LABELS[2],
        selection_weight=0.4,
        base_cluster_rate=7.5,
        energy_scale=55.0,
        eta_spread=0.95,
        forward_bias=0.25,
        channel_distribution=[
            ("central_jet", 0.36),
            ("forward_jet", 0.24),
            ("endcap_hadronic_shower", 0.18),
            ("tracker_lepton_track", 0.12),
            ("prompt_photon", 0.1),
        ],
    ),
}


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


def _select_profile() -> EventProfile:
    profiles = list(EVENT_PROFILES.values())
    weights = np.array([profile.selection_weight for profile in profiles], dtype=np.float64)
    weights /= weights.sum()
    index = np.random.choice(len(profiles), p=weights)
    return profiles[index]


def _select_channel(profile: EventProfile) -> SignalChannel:
    names, weights = zip(*profile.channel_distribution)
    probs = np.array(weights, dtype=np.float64)
    probs /= probs.sum()
    index = np.random.choice(len(names), p=probs)
    return SIGNAL_CHANNELS[names[index]]


def generate_particle(profile: EventProfile, geometry: DetectorGeometry) -> Particle:
    channel = _select_channel(profile)
    shape = 2.0 + 0.4 * channel.energy_modifier
    scale = max(profile.energy_scale * channel.energy_modifier / shape, 1e-3)
    energy = float(np.random.gamma(shape=shape, scale=scale))
    energy = max(energy, 1e-3)
    eta_width = max(0.25, channel.eta_spread * profile.eta_spread)
    orientation = 1.0 if random.random() > 0.5 else -1.0
    eta_center = channel.eta_center * orientation * (0.6 + profile.forward_bias)
    eta = float(np.random.normal(loc=eta_center, scale=eta_width))
    eta = float(np.clip(eta, -2.8, 2.8))
    phi = sample_phi()
    momentum = max(channel.momentum_fraction * energy, 1e-3)
    px = momentum * math.cos(phi)
    py = momentum * math.sin(phi)
    pz = momentum * math.sinh(eta)
    layer = choose_detector_layer(eta, geometry)
    four_vector = FourVector(energy=energy, px=px, py=py, pz=pz)
    return Particle(
        name=channel.name,
        four_vector=four_vector,
        eta=eta,
        phi=phi,
        detector_layer=layer,
        channel=channel,
    )


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
    event_name = event.event_family or "Çarpışma"

    return [
        CinematicPhase(
            name="Injection Sequencer",
            duration=0.35 * energy_scale,
            track_scale=0.15,
            glow_strength=0.1,
            annotation=f"{event_name}: Demetler hızlandırıcı tüneline besleniyor.",
        ),
        CinematicPhase(
            name="Magnetic Focusing",
            duration=0.45 * energy_scale,
            track_scale=0.35,
            glow_strength=0.25,
            annotation=f"Manyetik lensler {focus_layer} katmanında ışını odaklıyor.",
        ),
        CinematicPhase(
            name="Pre-Collision Drift",
            duration=0.60 * energy_scale,
            track_scale=0.65,
            glow_strength=0.4,
            annotation=f"{event_name}: Karşıt demetler hizalanıyor.",
        ),
        CinematicPhase(
            name="Collision Apex",
            duration=0.85 * energy_scale,
            track_scale=1.05,
            glow_strength=1.0,
            annotation=f"{event_name}: Enerji patlaması zirvede!",
        ),
        CinematicPhase(
            name="Cascade Afterglow",
            duration=0.55 * energy_scale,
            track_scale=1.2,
            glow_strength=0.6,
            annotation="Dedektör katmanları boyunca iz yağmuru.",
        ),
        CinematicPhase(
            name="Thermal Dissipation",
            duration=0.40 * energy_scale,
            track_scale=0.45,
            glow_strength=0.15,
            annotation="Kalorimetreler enerjiyi emiyor, sistem soğuyor.",
        ),
    ]


def generate_event(
    event_id: int,
    geometry: DetectorGeometry,
    *,
    max_particles: int = 11,
    transport: Optional[TransportSimulator] = None,
) -> CollisionEvent:
    profile = _select_profile()
    num_particles = int(np.random.poisson(lam=profile.base_cluster_rate)) + 1
    particles = [generate_particle(profile, geometry) for _ in range(num_particles)]
    features = build_feature_vector(particles, max_particles=max_particles)
    event = CollisionEvent(
        event_id=event_id,
        particles=particles,
        features=features,
        true_label=profile.label,
        event_family=profile.name,
    )
    event.cinematic_phases = build_cinematic_phases(event, geometry)
    if transport is not None:
        event.transport_summary = transport.propagate(particles)
    return event


def generate_event_stream(
    geometry: DetectorGeometry,
    *,
    max_events: Optional[int] = None,
    max_particles: int = 11,
    transport: Optional[TransportSimulator] = None,
) -> Generator[CollisionEvent, None, None]:
    event_id = 0
    if max_events is not None and max_events < 0:
        raise ValueError("max_events must be non-negative when provided")

    transport_engine = transport or TransportSimulator(geometry)

    while True:
        if max_events is not None and event_id >= max_events:
            break

        yield generate_event(
            event_id,
            geometry,
            max_particles=max_particles,
            transport=transport_engine,
        )
        event_id += 1
