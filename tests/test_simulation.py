"""Tests for the Kuantum simulation pipeline."""

from __future__ import annotations

import numpy as np

from kuantum.simulation.detector import default_geometry
from kuantum.simulation.physics import CollisionEvent, generate_event
from predict import load_predictor


def test_generate_event_feature_length():
    geometry = default_geometry()
    event = generate_event(0, geometry)
    assert isinstance(event, CollisionEvent)
    assert event.features.shape == (33,)


def test_predictor_outputs_integer_label():
    predictor = load_predictor()
    dummy_features = np.zeros(33, dtype=np.float32)
    label = predictor.predict(dummy_features)
    assert isinstance(label, int)
    assert 0 <= label < 3
