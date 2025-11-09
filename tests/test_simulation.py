"""Tests for the Kuantum simulation pipeline."""

from __future__ import annotations

import numpy as np

from kuantum.simulation.detector import default_geometry
from kuantum.simulation.physics import CollisionEvent, generate_event
from kuantum.simulation.main import EventLoop, SimulationConfig
from predict import load_predictor


def test_generate_event_feature_length():
    geometry = default_geometry()
    event = generate_event(0, geometry)
    assert isinstance(event, CollisionEvent)
    assert event.features.shape == (33,)
    assert event.cinematic_phases, "cinematic storyboard should be populated"
    assert all(phase.duration > 0 for phase in event.cinematic_phases)


def test_predictor_outputs_integer_label():
    predictor = load_predictor()
    dummy_features = np.zeros(33, dtype=np.float32)
    label = predictor.predict(dummy_features)
    assert isinstance(label, int)
    assert 0 <= label < 3


def test_timeline_preloads_events():
    geometry = default_geometry()
    config = SimulationConfig(
        max_events=None,
        event_rate=3.0,
        prediction_filter=None,
        visualize=False,
        simulation_duration=1.0,
        initial_speed=1.0,
    )
    loop = EventLoop(geometry, config)
    loop.prepare_timeline()
    assert loop.timeline, "timeline should not be empty"
    assert all(isinstance(frame.event, CollisionEvent) for frame in loop.timeline)


def test_zero_rate_requires_max_events():
    geometry = default_geometry()
    config = SimulationConfig(
        max_events=None,
        event_rate=0.0,
        prediction_filter=None,
        visualize=False,
        simulation_duration=5.0,
        initial_speed=1.0,
    )
    loop = EventLoop(geometry, config)
    try:
        loop.prepare_timeline()
    except ValueError as exc:
        assert "Event rate must be positive" in str(exc)
    else:
        raise AssertionError("Expected ValueError when event_rate <= 0 without max_events")
