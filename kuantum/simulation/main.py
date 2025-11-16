"""Command line interface for the Kuantum detector simulation."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Dict

import numpy as np

from ..models import load_model_and_scaler
from .analysis import MultiAgentAnalysisPanel
from .detector import DetectorGeometry, default_geometry
from .geometry_io import load_geometry
from .physics import CLASS_LABELS, CollisionEvent, generate_event_stream
from .reporting import SimulationReporter
from .transport import TransportSimulator
from .visualization import DetectorVisualizer
from .vr import VRSceneExporter


@dataclass
class SimulationConfig:
    max_events: Optional[int]
    event_rate: float
    prediction_filter: Optional[int]
    visualize: bool
    simulation_duration: float
    initial_speed: float
    vr_export_path: Optional[Path]
    report_path: Optional[Path]
    compile_report: bool

    @property
    def event_interval(self) -> float:
        return 0.0 if self.event_rate <= 0 else 1.0 / self.event_rate


@dataclass
class EventFrame:
    event: CollisionEvent
    time_offset: float


class PlaybackController:
    """Interactive playback controller for slow motion, pause, and quit."""

    def __init__(self, initial_speed: float = 1.0):
        import threading

        self.playback_speed = max(initial_speed, 0.125)
        self.paused = False
        self.should_stop = False
        self._bullet_time_request = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._input_loop, daemon=True)

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        self.should_stop = True

    def _input_loop(self) -> None:
        print(
            "[Controls] '+' hızlandır | '-' yavaşlat | 'p' duraklat/devam | '1' normal hız | 'b' çarpışma ağır çekim | 'q' çıkış"
        )
        print(
            "[Görselleştirme] Fare ile döndür/pan/zoom | Yukarı/Aşağı ok: kamera zoom | 'r': kamera sıfırla"
        )
        for line in sys.stdin:
            command = line.strip().lower()
            if not command:
                continue
            if command in {"q", "quit", "exit"}:
                self.should_stop = True
                print("[Playback] Çıkış isteniyor...")
                break
            if command in {"p", "pause"}:
                self.toggle_pause()
                continue
            if command in {"1", "normal"}:
                self.playback_speed = 1.0
                print("[Playback] Hız x1.00 olarak sıfırlandı")
                continue
            if command in {"+", "f", "faster"}:
                self.playback_speed = min(self.playback_speed * 2.0, 16.0)
                print(f"[Playback] Hız artırıldı: x{self.playback_speed:.2f}")
                continue
            if command in {"-", "s", "slower"}:
                self.playback_speed = max(self.playback_speed / 2.0, 0.125)
                print(f"[Playback] Hız azaltıldı: x{self.playback_speed:.2f}")
                continue
            if command in {"b", "bullet", "slowmo", "slow"}:
                self.request_bullet_time()
                continue
            print("[Playback] Komut anlaşılamadı.")

    def consume_bullet_time(self) -> bool:
        with self._lock:
            state = self._bullet_time_request
            self._bullet_time_request = False
            return state

    def request_bullet_time(self) -> None:
        with self._lock:
            self._bullet_time_request = True
        print("[Playback] Bir sonraki çarpışma ağır çekimde oynatılacak")

    def toggle_pause(self) -> None:
        with self._lock:
            self.paused = not self.paused
            paused = self.paused
        state = "DURAKLATILDI" if paused else "DEVAM"
        print(f"[Playback] {state}")


class EventLoop:
    def __init__(self, geometry: DetectorGeometry, config: SimulationConfig):
        self.geometry = geometry
        self.config = config
        artifacts = load_model_and_scaler()
        self.model = artifacts.model
        self.scaler = artifacts.scaler
        self.device = artifacts.device
        self.num_continuous = artifacts.config.num_continuous
        self.timeline: List[EventFrame] = []
        self.controller = PlaybackController(config.initial_speed)
        self.transport = TransportSimulator(self.geometry)
        self.analysis_panel = MultiAgentAnalysisPanel()
        self._vr_export_path = config.vr_export_path
        self._vr_exporter = VRSceneExporter(self.geometry) if config.vr_export_path else None
        self._report_path = config.report_path
        self._reporter = (
            SimulationReporter(self.geometry, compile_pdf=config.compile_report) if config.report_path else None
        )
        self._artifacts_exported = False
        self._visualizer: Optional[DetectorVisualizer] = None

    def prepare_timeline(self) -> None:
        if self.config.simulation_duration <= 0 and self.config.max_events is None:
            raise ValueError("Simulation duration or max events must be positive.")
        if self.config.event_rate <= 0 and self.config.max_events is None:
            raise ValueError(
                "Event rate must be positive when max events is not specified to avoid endless generation."
            )
        max_events = self._determine_event_count()
        interval = self.config.event_interval
        scheduled_index = 0
        print("Simülasyon verileri hazırlanıyor...")
        for event in generate_event_stream(self.geometry, max_events=max_events, transport=self.transport):
            prediction = self.predict_event(event)
            event.attach_prediction(prediction)
            if self.config.prediction_filter is not None and prediction != self.config.prediction_filter:
                continue
            time_offset = scheduled_index * interval if interval > 0 else 0.0
            self.timeline.append(EventFrame(event=event, time_offset=time_offset))
            scheduled_index += 1
        print(f"Hazırlanan kare sayısı: {len(self.timeline)}")
        self._export_artifacts()

    def _determine_event_count(self) -> Optional[int]:
        if self.config.max_events is not None:
            return self.config.max_events
        if self.config.event_interval == 0 or self.config.simulation_duration <= 0:
            return None
        return max(int(self.config.simulation_duration / self.config.event_interval), 1)

    def run(self) -> None:
        if not self.timeline:
            self.prepare_timeline()
        if not self.timeline:
            print("Gösterilecek etkinlik yok.")
            return
        visualizer = (
            DetectorVisualizer(self.geometry, show_overlay=True) if self.config.visualize else None
        )
        if visualizer and visualizer.is_available():
            self._visualizer = visualizer
            self._visualizer.bind_controller(self.controller)
            self._visualizer.initialize_scene()
        else:
            self._visualizer = None
        self.controller.start()
        last_tick = time.perf_counter()
        for frame in self.timeline:
            if self.controller.should_stop:
                break
            while self.controller.paused and not self.controller.should_stop:
                time.sleep(0.05)
                last_tick = time.perf_counter()
            interval = self.config.event_interval
            if interval > 0:
                scaled_interval = interval / max(self.controller.playback_speed, 0.125)
                elapsed = time.perf_counter() - last_tick
                remaining = scaled_interval - elapsed
                if remaining > 0:
                    time.sleep(remaining)
            last_tick = time.perf_counter()
            analysis_snapshot = self.analysis_panel.update(frame.event)
            self.display_event(frame.event, analysis_snapshot)
            bullet_time = self.controller.consume_bullet_time()
            if self._visualizer and self._visualizer.is_available():
                self._visualizer.set_analysis_summary(analysis_snapshot)
                self._visualizer.animate_event(
                    frame.event,
                    playback_speed=self.controller.playback_speed,
                    bullet_time=bullet_time,
                )
        if self._visualizer and self._visualizer.is_available():
            self._visualizer.finalize()
            self._visualizer = None
        self.controller.stop()

    def _export_artifacts(self) -> None:
        if self._artifacts_exported or not self.timeline:
            return
        events = [frame.event for frame in self.timeline]
        if self._vr_exporter and self._vr_export_path is not None:
            try:
                output = self._vr_exporter.export_html(events, self._vr_export_path)
                print(f"[Export] VR sahnesi kaydedildi: {output}")
            except Exception as exc:
                print(f"[Export] VR sahnesi oluşturulamadı: {exc}", file=sys.stderr)
        if self._reporter and self._report_path is not None:
            try:
                output = self._reporter.build_report(events, self._report_path)
                print(f"[Export] Rapor taslağı yazıldı: {output}")
            except Exception as exc:
                print(f"[Export] Rapor üretimi başarısız: {exc}", file=sys.stderr)
        self._artifacts_exported = True

    def predict_event(self, event: CollisionEvent) -> int:
        features = event.features
        if features.shape[0] != self.num_continuous:
            features = np.pad(features, (0, max(0, self.num_continuous - features.shape[0])), mode="constant")
            features = features[: self.num_continuous]
        scaled = self.scaler.transform(features.reshape(1, -1))
        x_cont = np.asarray(scaled, dtype=np.float32)
        import torch

        with torch.no_grad():
            tensor = torch.tensor(x_cont, device=self.device)
            x_categ = torch.empty((tensor.size(0), 0), dtype=torch.long, device=self.device)
            x_cont_enc = self.model.encode_continuous(tensor)
            x_categ_enc = self.model.encode_categorical(x_categ)
            logits = self.model(x_categ, tensor, x_categ_enc, x_cont_enc)
            pred = torch.argmax(logits, dim=-1).item()
        return int(pred)

    def display_event(self, event: CollisionEvent, analysis_snapshot: Optional[Dict[str, str]] = None) -> None:
        predicted_name = (
            CLASS_LABELS.get(event.model_prediction, str(event.model_prediction))
            if event.model_prediction is not None
            else "?"
        )
        family = event.event_family or "Çarpışma"
        truth_name = CLASS_LABELS.get(event.true_label, "Bilinmeyen")
        print(
            f"Event {event.event_id} -> sahne: {family} | model tahmini: {predicted_name} | gerçekçi profil: {truth_name}"
        )
        for particle in event.particles:
            print(
                f"  {particle.display_name:22s} | E={particle.four_vector.energy:7.2f} GeV | eta={particle.eta:+.2f} | layer={particle.detector_layer}"
            )
        if event.transport_summary and event.transport_summary.layer_totals:
            print("  [Katman Enerji Profili]")
            for layer, energy in event.transport_summary.layer_totals.items():
                print(f"    {layer}: {energy:.2f} GeV")
        if analysis_snapshot:
            print("[Analiz Paneli]")
            for title, description in analysis_snapshot.items():
                print(f"  {title}: {description}")
        print("-" * 60)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Kuantum detector simulation")
    parser.add_argument("--events", type=int, default=None, help="Override number of events (optional)")
    parser.add_argument("--duration", type=float, default=None, help="Simulation duration in seconds")
    parser.add_argument("--rate", type=float, default=2.0, help="Events per second during playback")
    parser.add_argument("--speed", type=float, default=1.0, help="Initial playback speed multiplier")
    parser.add_argument("--visualize", action="store_true", help="Enable 3D visualization with PyVista")
    parser.add_argument("--filter", type=int, default=None, help="Only display events predicted as the given class label")
    parser.add_argument("--geometry-file", type=str, default=None, help="External detector geometry (GDML/ROOT/JSON)")
    parser.add_argument("--geometry-format", type=str, choices=["gdml", "json", "root"], default=None, help="Explicit geometry format override")
    parser.add_argument("--geometry-tree", type=str, default=None, help="ROOT tree containing layer definitions")
    parser.add_argument("--export-vr", type=str, default=None, help="Export VR-ready HTML scene to this path")
    parser.add_argument("--report", type=str, default=None, help="Export LaTeX run report to this path")
    parser.add_argument("--compile-report", action="store_true", help="Compile the LaTeX report with pdflatex")
    return parser.parse_args(argv)


def _prompt_for_duration() -> float:
    while True:
        try:
            value = float(input("Simülasyon süresi (s): "))
            if value <= 0:
                print("Lütfen sıfırdan büyük bir değer girin.")
                continue
            return value
        except ValueError:
            print("Geçerli bir sayı girin.")


def run_cli(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    duration = args.duration if args.duration is not None else _prompt_for_duration()
    if args.rate <= 0 and args.events is None:
        print(
            "Hata: Sonsuz olay üretimini önlemek için --rate pozitif olmalı ya da --events belirtilmeli.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if args.geometry_file:
        try:
            geometry = load_geometry(args.geometry_file, fmt=args.geometry_format, tree=args.geometry_tree)
        except Exception as exc:
            print(f"Geometri yüklenemedi: {exc}", file=sys.stderr)
            raise SystemExit(2)
    else:
        geometry = default_geometry()
    vr_path = Path(args.export_vr) if args.export_vr else None
    report_path = Path(args.report) if args.report else None
    config = SimulationConfig(
        max_events=args.events,
        event_rate=args.rate,
        prediction_filter=args.filter,
        visualize=args.visualize,
        simulation_duration=duration,
        initial_speed=args.speed,
        vr_export_path=vr_path,
        report_path=report_path,
        compile_report=args.compile_report,
    )
    loop = EventLoop(geometry, config)
    try:
        loop.prepare_timeline()
        print("Simülasyon başlatılıyor...")
        loop.run()
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")


if __name__ == "__main__":
    run_cli(sys.argv[1:])
