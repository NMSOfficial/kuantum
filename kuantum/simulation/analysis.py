"""Analysis agents and summary panel for collision events."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, MutableMapping

from .physics import CLASS_LABELS, CollisionEvent


class AnalysisAgent:
    name: str

    def update(self, event: CollisionEvent) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def snapshot(self) -> Dict[str, str]:  # pragma: no cover - interface
        raise NotImplementedError

    def reset(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class CrossSectionAgent(AnalysisAgent):
    name: str = "Cross Sections"
    _counts: MutableMapping[int, int] = field(default_factory=lambda: OrderedDict((i, 0) for i in sorted(CLASS_LABELS)))
    _total_events: int = 0

    def update(self, event: CollisionEvent) -> None:
        label = event.model_prediction
        if label is None:
            return
        self._counts[label] = self._counts.get(label, 0) + 1
        self._total_events += 1

    def snapshot(self) -> Dict[str, str]:
        if self._total_events == 0:
            return {self.name: "No predictions"}
        lines = []
        for label, count in self._counts.items():
            fraction = count / max(self._total_events, 1)
            lines.append(f"{CLASS_LABELS.get(label, label)}: {count} ({fraction:.1%})")
        return {self.name: " | ".join(lines)}

    def reset(self) -> None:
        for key in list(self._counts.keys()):
            self._counts[key] = 0
        self._total_events = 0


@dataclass
class TransportDiagnosticsAgent(AnalysisAgent):
    name: str = "Layer Energy"
    _totals: MutableMapping[str, float] = field(default_factory=OrderedDict)

    def update(self, event: CollisionEvent) -> None:
        if event.transport_summary is None:
            return
        for layer, energy in event.transport_summary.layer_totals.items():
            self._totals[layer] = self._totals.get(layer, 0.0) + energy

    def snapshot(self) -> Dict[str, str]:
        if not self._totals:
            return {self.name: "No transport data"}
        lines = [f"{layer}: {energy:.1f} GeV" for layer, energy in self._totals.items()]
        return {self.name: " | ".join(lines)}

    def reset(self) -> None:
        self._totals.clear()


@dataclass
class AnomalyAgent(AnalysisAgent):
    name: str = "Anomaly Watch"
    _mismatches: List[int] = field(default_factory=list)

    def update(self, event: CollisionEvent) -> None:
        if event.model_prediction is None or event.true_label is None:
            return
        if event.model_prediction != event.true_label:
            self._mismatches.append(event.event_id)

    def snapshot(self) -> Dict[str, str]:
        if not self._mismatches:
            return {self.name: "No discrepancies"}
        latest = ", ".join(map(str, self._mismatches[-5:]))
        return {self.name: f"Mismatch events: {latest}"}

    def reset(self) -> None:
        self._mismatches.clear()


@dataclass
class MultiAgentAnalysisPanel:
    agents: List[AnalysisAgent] = field(default_factory=lambda: [CrossSectionAgent(), TransportDiagnosticsAgent(), AnomalyAgent()])
    _latest_snapshot: Dict[str, str] = field(default_factory=dict, init=False)

    def update(self, event: CollisionEvent) -> Dict[str, str]:
        for agent in self.agents:
            agent.update(event)
        self._latest_snapshot = self.snapshot()
        return self._latest_snapshot

    def snapshot(self) -> Dict[str, str]:
        snapshot: Dict[str, str] = OrderedDict()
        for agent in self.agents:
            snapshot.update(agent.snapshot())
        return snapshot

    def reset(self) -> None:
        for agent in self.agents:
            agent.reset()
        self._latest_snapshot = {}


__all__ = [
    "AnalysisAgent",
    "CrossSectionAgent",
    "TransportDiagnosticsAgent",
    "AnomalyAgent",
    "MultiAgentAnalysisPanel",
]
