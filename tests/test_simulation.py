"""Tests for the Kuantum simulation pipeline."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from kuantum.simulation.analysis import MultiAgentAnalysisPanel
from kuantum.simulation.detector import default_geometry
from kuantum.simulation.geometry_io import load_geometry
from kuantum.simulation.main import EventLoop, SimulationConfig
from kuantum.simulation.physics import CollisionEvent, generate_event
from kuantum.simulation.reporting import SimulationReporter
from kuantum.simulation.transport import TransportSimulator
from kuantum.simulation.vr import VRSceneExporter
from predict import load_predictor


def test_generate_event_feature_length():
    geometry = default_geometry()
    event = generate_event(0, geometry)
    assert isinstance(event, CollisionEvent)
    assert event.features.shape == (33,)
    assert event.cinematic_phases, "cinematic storyboard should be populated"
    assert all(phase.duration > 0 for phase in event.cinematic_phases)
    assert event.true_label in {0, 1, 2}
    assert event.event_family


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
        vr_export_path=None,
        report_path=None,
        compile_report=False,
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
        vr_export_path=None,
        report_path=None,
        compile_report=False,
    )
    loop = EventLoop(geometry, config)
    try:
        loop.prepare_timeline()
    except ValueError as exc:
        assert "Event rate must be positive" in str(exc)
    else:
        raise AssertionError("Expected ValueError when event_rate <= 0 without max_events")

def test_geometry_loader_gdml(tmp_path):
    gdml = tmp_path / "detector.gdml"
    gdml.write_text(
        """<?xml version='1.0'?>
<gdml>
  <solids>
    <tube name='Tracker' rmax='2000' rmin='0' z='4000' lunit='mm' />
  </solids>
  <structure>
    <volume name='InnerTracker'>
      <materialref ref='Silicon'/>
      <solidref ref='Tracker'/>
    </volume>
  </structure>
</gdml>
""",
        encoding="utf-8",
    )
    geometry = load_geometry(gdml, fmt="gdml")
    assert len(geometry.layers) == 1
    layer = geometry.layers[0]
    assert layer.inner_radius == 0
    assert layer.outer_radius == 2.0  # converted to metres


def _sample_event_with_transport():
    geometry = default_geometry()
    transport = TransportSimulator(geometry)
    event = generate_event(0, geometry, transport=transport)
    event.attach_prediction(event.true_label or 0)
    return event


def test_transport_summary_attached():
    event = _sample_event_with_transport()
    assert event.transport_summary is not None
    assert event.transport_summary.layer_totals


def test_analysis_panel_tracks_counts():
    panel = MultiAgentAnalysisPanel()
    event = _sample_event_with_transport()
    snapshot = panel.update(event)
    assert "Cross Sections" in snapshot


def test_vr_exporter_produces_html(tmp_path):
    event = _sample_event_with_transport()
    exporter = VRSceneExporter(default_geometry())
    html_path = tmp_path / "scene.html"
    exporter.export_html([event], html_path)
    assert html_path.exists()
    contents = html_path.read_text(encoding="utf-8")
    assert "three" in contents.lower()


def test_simulation_reporter_outputs_tex(tmp_path):
    event = _sample_event_with_transport()
    reporter = SimulationReporter(default_geometry())
    tex_path = tmp_path / "report.tex"
    reporter.build_report([event], tex_path)
    assert tex_path.exists()
    content = tex_path.read_text(encoding="utf-8")
    assert "\documentclass" in content
